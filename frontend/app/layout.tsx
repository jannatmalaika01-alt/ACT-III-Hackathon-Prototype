import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Factory Quality & Maintenance Agent",
  description: "Detect defects, find the fix in SOPs, approve, create the task.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}