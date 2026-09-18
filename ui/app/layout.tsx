import type { Metadata } from "next";
import Link from "next/link";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Policy QA — grounded retrieval",
  description:
    "Answers quoted verbatim from the Employee Expense Policy, or an abstention.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="flex h-full min-h-full flex-col overflow-hidden bg-base">
        <header className="flex shrink-0 items-center gap-4 border-b border-hairline px-6 py-3">
          <span className="text-[13px] tracking-[-0.01em] text-ink">
            Policy&nbsp;QA
          </span>
          <span className="num text-[10px] tracking-[0.08em] text-ink-faint uppercase">
            grounded retrieval
          </span>
          <nav className="ml-auto flex items-center gap-5">
            <Link
              href="/"
              className="text-[11px] tracking-[0.08em] text-ink-dim uppercase hover:text-ink"
            >
              Ask
            </Link>
            <Link
              href="/evals"
              className="text-[11px] tracking-[0.08em] text-ink-dim uppercase hover:text-ink"
            >
              Evals
            </Link>
          </nav>
        </header>
        <main className="min-h-0 flex-1">{children}</main>
      </body>
    </html>
  );
}
