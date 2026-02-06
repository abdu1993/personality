import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'AI Capex Efficiency Dashboard',
  description: 'Track whether hyperscaler AI capex translates into real, utilized compute.',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className={inter.className}>
        <div className="min-h-screen">
          <header className="border-b border-[#2a2a3a] px-6 py-4">
            <div className="max-w-[1600px] mx-auto flex items-center justify-between">
              <div>
                <h1 className="text-xl font-bold text-white">AI Capex Efficiency Dashboard</h1>
                <p className="text-sm text-zinc-400">Hyperscaler utilization signals</p>
              </div>
              <div className="flex gap-3 items-center">
                <span className="text-xs text-zinc-500">Auto-refresh: 5min</span>
                <span className="inline-block w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
              </div>
            </div>
          </header>
          <main className="max-w-[1600px] mx-auto px-6 py-6">
            {children}
          </main>
        </div>
      </body>
    </html>
  )
}
