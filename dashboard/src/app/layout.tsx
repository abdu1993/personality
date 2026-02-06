import type { Metadata } from 'next'
import './globals.css'

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
      <body className="font-sans">
        <div className="min-h-screen">
          <header className="border-b border-[#2a2a3a] px-6 py-4 sticky top-0 z-50 backdrop-blur-md bg-[#0a0a0f]/80">
            <div className="max-w-[1600px] mx-auto flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white text-sm font-bold">
                  AI
                </div>
                <div>
                  <h1 className="text-lg font-bold text-white">AI Capex Efficiency Dashboard</h1>
                  <p className="text-xs text-zinc-500">Hyperscaler utilization confidence signals</p>
                </div>
              </div>
              <div className="flex gap-4 items-center">
                <span className="text-xs text-zinc-600 hidden sm:inline">Demo Mode — Static Data</span>
                <span className="inline-flex items-center gap-1.5 text-xs text-emerald-500">
                  <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
                  Live
                </span>
              </div>
            </div>
          </header>
          <main className="max-w-[1600px] mx-auto px-4 sm:px-6 py-6">
            {children}
          </main>
        </div>
      </body>
    </html>
  )
}
