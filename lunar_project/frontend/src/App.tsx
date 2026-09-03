import "./globals.css";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning={true}>
      <body>{children}</body>
    </html>
  );
}

export default function App() {
  return (
    <div className="min-h-screen bg-slate-950 text-cyan-100">
    </div>
  );
}