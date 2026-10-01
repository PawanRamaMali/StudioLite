import type { Metadata } from "next";
import { connection } from "next/server";
import { Geist, Geist_Mono } from "next/font/google";
import { TOKEN_META_NAME } from "@/lib/api";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "StudioLite - AI Video Studio",
  description: "AI Video Generation & Editing Studio",
};

export default async function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  // The launchers pass the backend's .auth token as STUDIOLITE_API_TOKEN.
  // Render per request so a prebuilt (`next start`) bundle still gets the
  // token of the backend it runs next to; getApiToken() reads the meta tag.
  await connection();
  const apiToken = process.env.STUDIOLITE_API_TOKEN || "";
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased dark`}>
      <head>{apiToken && <meta name={TOKEN_META_NAME} content={apiToken} />}</head>
      <body className="min-h-full bg-[#09090b] text-[#fafafa]">{children}</body>
    </html>
  );
}
