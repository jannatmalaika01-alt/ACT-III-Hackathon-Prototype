-- =====================================================================
-- AI Factory Quality & Maintenance Agent - database schema
-- Run in: Supabase Dashboard -> SQL Editor -> New query -> Run
-- =====================================================================

-- ---------- Enums ----------
create type public.case_status as enum
  ('pending', 'approved', 'rejected', 'task_created', 'task_failed');

create type public.approval_decision as enum ('approved', 'rejected');

create type public.severity_level as enum ('low', 'medium', 'high', 'critical');

-- ---------- cases ----------
create table public.cases (
  id                 uuid primary key default gen_random_uuid(),
  status             public.case_status not null default 'pending',

  -- uploaded input
  file_path          text not null,          -- path inside the storage bucket
  file_name          text not null,
  content_type       text,
  uploaded_by        text,

  -- diagnosis (filled in by the AI pipeline after upload)
  defect             text,
  severity           public.severity_level,
  explanation        text,
  recommended_action text,
  source_manual      text,                   -- citation: manual name
  source_page        integer check (source_page > 0),  -- citation: page number
  confidence         numeric(4,3) check (confidence between 0 and 1),
  diagnosis_raw      jsonb,                  -- full JSON from the model, for debugging
  diagnosed_at       timestamptz,

  -- Evolus task result
  evolus_task_id     text,
  task_error         text,
  task_attempts      integer not null default 0,

  created_at         timestamptz not null default now(),
  updated_at         timestamptz not null default now(),

  -- integrity rules enforced by the database itself
  constraint task_created_has_id
    check (status <> 'task_created' or evolus_task_id is not null),
  constraint task_failed_has_error
    check (status <> 'task_failed' or task_error is not null)
);

create index cases_status_created_idx on public.cases (status, created_at desc);

-- ---------- approvals (one final human decision per case) ----------
create table public.approvals (
  id          uuid primary key default gen_random_uuid(),
  case_id     uuid not null references public.cases(id) on delete cascade,
  decision    public.approval_decision not null,
  decided_by  text not null,
  reason      text,
  decided_at  timestamptz not null default now(),

  constraint rejection_needs_reason
    check (decision <> 'rejected' or (reason is not null and length(trim(reason)) > 0))
);

create unique index approvals_one_decision_per_case on public.approvals (case_id);

-- ---------- case_events (audit trail + webhook de-duplication) ----------
create table public.case_events (
  id           bigint generated always as identity primary key,
  case_id      uuid not null references public.cases(id) on delete cascade,
  event_id     text unique,                  -- webhook event id; NULL for internal events
  source       text not null default 'api',  -- api | webhook
  from_status  public.case_status,
  to_status    public.case_status not null,
  note         text,
  created_at   timestamptz not null default now()
);

create index case_events_case_idx on public.case_events (case_id, created_at);

-- ---------- keep updated_at fresh ----------
create or replace function public.set_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end $$;

create trigger cases_set_updated_at
  before update on public.cases
  for each row execute function public.set_updated_at();

-- ---------- security ----------
-- RLS on with no policies = only the backend (service_role key) can touch data.
alter table public.cases       enable row level security;
alter table public.approvals   enable row level security;
alter table public.case_events enable row level security;

-- ---------- private storage bucket for uploads ----------
insert into storage.buckets (id, name, public)
values ('case-uploads', 'case-uploads', false)
on conflict (id) do nothing;
