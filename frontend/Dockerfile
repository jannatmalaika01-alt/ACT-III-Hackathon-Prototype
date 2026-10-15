# Next.js standalone build. BACKEND_URL and NEXT_PUBLIC_* are baked in at BUILD time (see docker-compose.yml).
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
ARG BACKEND_URL=http://backend:8000
ARG NEXT_PUBLIC_USE_MOCK=false
ARG NEXT_PUBLIC_MODEL_LABEL=
ARG NEXT_PUBLIC_EVOLUS_URL=
ENV BACKEND_URL=$BACKEND_URL NEXT_PUBLIC_USE_MOCK=$NEXT_PUBLIC_USE_MOCK \
    NEXT_PUBLIC_MODEL_LABEL=$NEXT_PUBLIC_MODEL_LABEL NEXT_PUBLIC_EVOLUS_URL=$NEXT_PUBLIC_EVOLUS_URL
RUN npm run build

FROM node:22-alpine
WORKDIR /app
ENV NODE_ENV=production PORT=3000 HOSTNAME=0.0.0.0
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
COPY --from=build /app/public ./public
EXPOSE 3000
CMD ["node", "server.js"]
