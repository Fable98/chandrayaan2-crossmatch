import type { Metadata } from "next";
import { Inter, IBM_Plex_Sans, IBM_Plex_Mono, Playfair_Display } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
  variable: "--font-inter",
});

const playfair = Playfair_Display({
  subsets: ["latin"],
  style: ["normal", "italic"],
  weight: ["400", "500", "600"],
  variable: "--font-playfair",
});

const plexSans = IBM_Plex_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-plex-sans",
});

const plexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-plex-mono",
});

export const metadata: Metadata = {
  title: "Astralynx — Step Into the Celestial Frontier",
  description:
    "Discover the fusion of cosmic exploration and intelligent innovation, where every step unveils a universe powered by AI.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${playfair.variable} ${plexSans.variable} ${plexMono.variable}`}
    >
      <body className="bg-[#070d14] text-[#f0f4f5] font-sans antialiased">
        {children}
      </body>
    </html>
  );
}
