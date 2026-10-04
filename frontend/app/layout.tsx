import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import { CrossFilterProvider } from "@/components/crossfilter";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: {
    default: "ROADRISK Intelligence - Road Collision Severity Prediction & Road Safety Analytics",
    template: "%s | ROADRISK Intelligence",
  },
  description:
    "Machine-learning powered road safety intelligence built on UK Department for Transport STATS19 collision data (2021-2025).",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">
        <Navbar />
        <main className="flex-1">
          <CrossFilterProvider>{children}</CrossFilterProvider>
        </main>
        <Footer />
      </body>
    </html>
  );
}
