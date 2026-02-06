// Main React Application
const { useState, useEffect, useRef } = React;

// Utility functions
const formatCurrency = (num) => {
    if (num >= 1e12) return `$${(num / 1e12).toFixed(1)}T`;
    if (num >= 1e9) return `$${(num / 1e9).toFixed(1)}B`;
    if (num >= 1e6) return `$${(num / 1e6).toFixed(1)}M`;
    return `$${num.toLocaleString()}`;
};

const formatPercent = (num, decimals = 2) => {
    if (num === null || num === undefined) return 'N/A';
    return `${num >= 0 ? '+' : ''}${num.toFixed(decimals)}%`;
};

// Header Component
const Header = () => (
    <header className="gradient-bg text-white py-8 px-6">
        <div className="max-w-7xl mx-auto">
            <h1 className="text-3xl font-bold mb-2">ETF Comparison Tool</h1>
            <p className="text-blue-200 text-lg">Compare Halal ETFs, ESG Funds, and Index Funds</p>
            <p className="text-blue-300 text-sm mt-2">
                Analyzing performance, costs, and exclusions to understand the true cost of values-based investing
            </p>
        </div>
    </header>
);

// Navigation Tabs
const TabNavigation = ({ activeTab, setActiveTab }) => {
    const tabs = [
        { id: 'overview', label: 'Overview' },
        { id: 'performance', label: 'Performance' },
        { id: 'costs', label: 'Costs & Fees' },
        { id: 'exclusions', label: 'Exclusions' },
        { id: 'backtest', label: 'Backtester' },
        { id: 'holdings', label: 'Holdings' }
    ];

    return (
        <nav className="bg-white border-b border-gray-200 sticky top-0 z-10">
            <div className="max-w-7xl mx-auto px-6">
                <div className="flex space-x-8 overflow-x-auto">
                    {tabs.map(tab => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            className={`py-4 px-2 font-medium text-sm whitespace-nowrap transition-colors
                                ${activeTab === tab.id
                                    ? 'tab-active'
                                    : 'text-gray-500 hover:text-gray-700'}`}
                        >
                            {tab.label}
                        </button>
                    ))}
                </div>
            </div>
        </nav>
    );
};

// Stat Card Component
const StatCard = ({ title, value, subtitle, trend, color = 'blue' }) => {
    const colors = {
        blue: 'bg-blue-50 border-blue-200',
        green: 'bg-green-50 border-green-200',
        red: 'bg-red-50 border-red-200',
        yellow: 'bg-yellow-50 border-yellow-200',
        purple: 'bg-purple-50 border-purple-200'
    };

    return (
        <div className={`stat-card p-5 rounded-xl border ${colors[color]} card-shadow`}>
            <p className="text-gray-600 text-sm font-medium">{title}</p>
            <p className="text-2xl font-bold text-gray-800 mt-1">{value}</p>
            {subtitle && <p className="text-sm text-gray-500 mt-1">{subtitle}</p>}
            {trend && (
                <p className={`text-sm mt-2 ${trend >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {trend >= 0 ? '↑' : '↓'} {Math.abs(trend).toFixed(2)}%
                </p>
            )}
        </div>
    );
};

// Overview Section
const OverviewSection = () => {
    const allETFs = [...ETF_DATA.index, ...ETF_DATA.halal, ...ETF_DATA.esg];

    return (
        <div className="fade-in space-y-8">
            {/* Key Insights Banner */}
            <div className="bg-gradient-to-r from-red-50 to-orange-50 border border-red-200 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-red-800 mb-3">Key Findings</h3>
                <div className="grid md:grid-cols-3 gap-4 text-sm">
                    <div className="flex items-start space-x-2">
                        <span className="text-red-500 text-lg">1.</span>
                        <p className="text-gray-700">
                            <strong>Halal ETFs cost 10-18x more</strong> than index funds (0.49-0.65% vs 0.03%)
                        </p>
                    </div>
                    <div className="flex items-start space-x-2">
                        <span className="text-red-500 text-lg">2.</span>
                        <p className="text-gray-700">
                            <strong>53% of S&P 500 excluded</strong> from Halal funds due to debt/interest screens
                        </p>
                    </div>
                    <div className="flex items-start space-x-2">
                        <span className="text-red-500 text-lg">3.</span>
                        <p className="text-gray-700">
                            <strong>Historical underperformance</strong> of 1-3% annually vs broad market
                        </p>
                    </div>
                </div>
            </div>

            {/* Summary Stats by Category */}
            <div className="grid md:grid-cols-3 gap-6">
                {/* Index Funds */}
                <div className="bg-white rounded-xl p-6 card-shadow border-t-4 border-gray-500">
                    <h3 className="font-semibold text-lg text-gray-800 mb-4">Index Funds (Benchmark)</h3>
                    <div className="space-y-3">
                        <div className="flex justify-between">
                            <span className="text-gray-600">Avg Expense Ratio</span>
                            <span className="font-semibold text-green-600">0.05%</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-600">Holdings</span>
                            <span className="font-semibold">~503</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-600">Market Coverage</span>
                            <span className="font-semibold">100%</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-600">Total AUM</span>
                            <span className="font-semibold">$1.2T+</span>
                        </div>
                    </div>
                </div>

                {/* Halal Funds */}
                <div className="bg-white rounded-xl p-6 card-shadow border-t-4 border-green-500">
                    <h3 className="font-semibold text-lg text-gray-800 mb-4">Halal ETFs</h3>
                    <div className="space-y-3">
                        <div className="flex justify-between">
                            <span className="text-gray-600">Avg Expense Ratio</span>
                            <span className="font-semibold text-red-600">0.55%</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-600">Holdings</span>
                            <span className="font-semibold">~239</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-600">Market Coverage</span>
                            <span className="font-semibold text-orange-600">47%</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-600">Total AUM</span>
                            <span className="font-semibold">~$1B</span>
                        </div>
                    </div>
                    <div className="mt-4 pt-4 border-t">
                        <p className="text-sm text-red-600 font-medium">
                            Excludes 53% of S&P 500 companies
                        </p>
                    </div>
                </div>

                {/* ESG Funds */}
                <div className="bg-white rounded-xl p-6 card-shadow border-t-4 border-blue-500">
                    <h3 className="font-semibold text-lg text-gray-800 mb-4">ESG Funds</h3>
                    <div className="space-y-3">
                        <div className="flex justify-between">
                            <span className="text-gray-600">Avg Expense Ratio</span>
                            <span className="font-semibold text-yellow-600">0.15%</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-600">Holdings</span>
                            <span className="font-semibold">~528</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-600">Market Coverage</span>
                            <span className="font-semibold">85%</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-600">Total AUM</span>
                            <span className="font-semibold">$27B+</span>
                        </div>
                    </div>
                    <div className="mt-4 pt-4 border-t">
                        <p className="text-sm text-orange-600 font-medium">
                            Excludes 15% based on ESG criteria
                        </p>
                    </div>
                </div>
            </div>

            {/* All ETFs Table */}
            <div className="bg-white rounded-xl card-shadow overflow-hidden">
                <div className="p-6 border-b">
                    <h3 className="font-semibold text-lg">All ETFs Comparison</h3>
                </div>
                <div className="overflow-x-auto">
                    <table className="w-full">
                        <thead className="bg-gray-50">
                            <tr>
                                <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase">Ticker</th>
                                <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase">Name</th>
                                <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase">Type</th>
                                <th className="px-6 py-3 text-right text-xs font-semibold text-gray-600 uppercase">Expense Ratio</th>
                                <th className="px-6 py-3 text-right text-xs font-semibold text-gray-600 uppercase">Holdings</th>
                                <th className="px-6 py-3 text-right text-xs font-semibold text-gray-600 uppercase">AUM</th>
                                <th className="px-6 py-3 text-right text-xs font-semibold text-gray-600 uppercase">2024 Return</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-200">
                            {allETFs.map(etf => {
                                const return2024 = PERFORMANCE_DATA[etf.ticker]?.[9];
                                const typeColors = {
                                    'Index': 'bg-gray-100 text-gray-800',
                                    'Halal': 'bg-green-100 text-green-800',
                                    'ESG': 'bg-blue-100 text-blue-800'
                                };
                                return (
                                    <tr key={etf.ticker} className="hover:bg-gray-50">
                                        <td className="px-6 py-4 font-semibold text-blue-600">{etf.ticker}</td>
                                        <td className="px-6 py-4 text-sm text-gray-700">{etf.name}</td>
                                        <td className="px-6 py-4">
                                            <span className={`px-2 py-1 rounded-full text-xs font-medium ${typeColors[etf.type]}`}>
                                                {etf.type}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4 text-right">
                                            <span className={etf.expenseRatio > 0.2 ? 'text-red-600 font-medium' : 'text-green-600'}>
                                                {etf.expenseRatio.toFixed(2)}%
                                            </span>
                                        </td>
                                        <td className="px-6 py-4 text-right text-gray-700">{etf.holdings}</td>
                                        <td className="px-6 py-4 text-right text-gray-700">{formatCurrency(etf.aum)}</td>
                                        <td className="px-6 py-4 text-right">
                                            <span className={return2024 >= 0 ? 'text-green-600' : 'text-red-600'}>
                                                {formatPercent(return2024)}
                                            </span>
                                        </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};

// Performance Section with Chart
const PerformanceSection = () => {
    const chartRef = useRef(null);
    const chartInstance = useRef(null);
    const [selectedFunds, setSelectedFunds] = useState(['SPY', 'SPUS', 'ESGU']);
    const [timeRange, setTimeRange] = useState('5Y');

    useEffect(() => {
        if (chartInstance.current) {
            chartInstance.current.destroy();
        }

        const ctx = chartRef.current.getContext('2d');

        const colors = {
            SPY: { line: '#374151', bg: 'rgba(55, 65, 81, 0.1)' },
            VOO: { line: '#6b7280', bg: 'rgba(107, 114, 128, 0.1)' },
            SPUS: { line: '#10b981', bg: 'rgba(16, 185, 129, 0.1)' },
            HLAL: { line: '#059669', bg: 'rgba(5, 150, 105, 0.1)' },
            ESGU: { line: '#3b82f6', bg: 'rgba(59, 130, 246, 0.1)' },
            SUSA: { line: '#2563eb', bg: 'rgba(37, 99, 235, 0.1)' },
            ESGV: { line: '#1d4ed8', bg: 'rgba(29, 78, 216, 0.1)' }
        };

        // Calculate cumulative returns
        const years = PERFORMANCE_DATA.years;
        const startIdx = timeRange === '5Y' ? 5 : timeRange === '3Y' ? 7 : 0;

        const datasets = selectedFunds.map(ticker => {
            const returns = PERFORMANCE_DATA[ticker].slice(startIdx);
            let cumulative = [100];
            returns.forEach((ret, i) => {
                if (ret !== null) {
                    cumulative.push(cumulative[cumulative.length - 1] * (1 + ret / 100));
                } else {
                    cumulative.push(null);
                }
            });

            return {
                label: ticker,
                data: cumulative.slice(1),
                borderColor: colors[ticker]?.line || '#888',
                backgroundColor: colors[ticker]?.bg || 'rgba(136, 136, 136, 0.1)',
                borderWidth: 2,
                fill: false,
                tension: 0.3,
                pointRadius: 4,
                pointHoverRadius: 6
            };
        });

        chartInstance.current = new Chart(ctx, {
            type: 'line',
            data: {
                labels: years.slice(startIdx),
                datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { usePointStyle: true, padding: 20 }
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const value = context.parsed.y;
                                const change = value - 100;
                                return `${context.dataset.label}: $${value.toFixed(2)} (${change >= 0 ? '+' : ''}${change.toFixed(2)}%)`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        title: { display: true, text: 'Value of $100 Investment' },
                        ticks: { callback: (value) => '$' + value }
                    },
                    x: {
                        title: { display: true, text: 'Year' }
                    }
                }
            }
        });

        return () => {
            if (chartInstance.current) {
                chartInstance.current.destroy();
            }
        };
    }, [selectedFunds, timeRange]);

    const allTickers = ['SPY', 'VOO', 'SPUS', 'HLAL', 'ESGU', 'SUSA', 'ESGV'];

    const toggleFund = (ticker) => {
        if (selectedFunds.includes(ticker)) {
            if (selectedFunds.length > 1) {
                setSelectedFunds(selectedFunds.filter(t => t !== ticker));
            }
        } else {
            setSelectedFunds([...selectedFunds, ticker]);
        }
    };

    // Calculate performance stats
    const getStats = (ticker) => {
        const returns = PERFORMANCE_DATA[ticker].filter(r => r !== null);
        const avg = returns.reduce((a, b) => a + b, 0) / returns.length;
        const best = Math.max(...returns);
        const worst = Math.min(...returns);
        return { avg, best, worst };
    };

    return (
        <div className="fade-in space-y-6">
            {/* Controls */}
            <div className="bg-white rounded-xl p-6 card-shadow">
                <div className="flex flex-wrap items-center justify-between gap-4">
                    <div>
                        <h3 className="font-semibold text-gray-800 mb-2">Select ETFs to Compare</h3>
                        <div className="flex flex-wrap gap-2">
                            {allTickers.map(ticker => (
                                <button
                                    key={ticker}
                                    onClick={() => toggleFund(ticker)}
                                    className={`px-3 py-1.5 rounded-full text-sm font-medium transition-colors
                                        ${selectedFunds.includes(ticker)
                                            ? 'bg-blue-600 text-white'
                                            : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}
                                >
                                    {ticker}
                                </button>
                            ))}
                        </div>
                    </div>
                    <div>
                        <h3 className="font-semibold text-gray-800 mb-2">Time Range</h3>
                        <div className="flex gap-2">
                            {['3Y', '5Y', 'ALL'].map(range => (
                                <button
                                    key={range}
                                    onClick={() => setTimeRange(range)}
                                    className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors
                                        ${timeRange === range
                                            ? 'bg-gray-800 text-white'
                                            : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}
                                >
                                    {range}
                                </button>
                            ))}
                        </div>
                    </div>
                </div>
            </div>

            {/* Chart */}
            <div className="bg-white rounded-xl p-6 card-shadow">
                <h3 className="font-semibold text-lg mb-4">Cumulative Performance ($100 Investment)</h3>
                <div className="chart-container">
                    <canvas ref={chartRef}></canvas>
                </div>
            </div>

            {/* Performance Stats Table */}
            <div className="bg-white rounded-xl card-shadow overflow-hidden">
                <div className="p-6 border-b">
                    <h3 className="font-semibold text-lg">Annual Return Statistics</h3>
                </div>
                <div className="overflow-x-auto">
                    <table className="w-full">
                        <thead className="bg-gray-50">
                            <tr>
                                <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase">ETF</th>
                                <th className="px-6 py-3 text-right text-xs font-semibold text-gray-600 uppercase">Avg Annual</th>
                                <th className="px-6 py-3 text-right text-xs font-semibold text-gray-600 uppercase">Best Year</th>
                                <th className="px-6 py-3 text-right text-xs font-semibold text-gray-600 uppercase">Worst Year</th>
                                <th className="px-6 py-3 text-right text-xs font-semibold text-gray-600 uppercase">vs SPY (Avg)</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-200">
                            {selectedFunds.map(ticker => {
                                const stats = getStats(ticker);
                                const spyStats = getStats('SPY');
                                const vsSpy = stats.avg - spyStats.avg;
                                return (
                                    <tr key={ticker} className="hover:bg-gray-50">
                                        <td className="px-6 py-4 font-semibold">{ticker}</td>
                                        <td className="px-6 py-4 text-right">{formatPercent(stats.avg)}</td>
                                        <td className="px-6 py-4 text-right text-green-600">{formatPercent(stats.best)}</td>
                                        <td className="px-6 py-4 text-right text-red-600">{formatPercent(stats.worst)}</td>
                                        <td className="px-6 py-4 text-right">
                                            <span className={vsSpy >= 0 ? 'text-green-600' : 'text-red-600'}>
                                                {formatPercent(vsSpy)}
                                            </span>
                                        </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Year by Year Returns */}
            <div className="bg-white rounded-xl card-shadow overflow-hidden">
                <div className="p-6 border-b">
                    <h3 className="font-semibold text-lg">Year-by-Year Returns</h3>
                </div>
                <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                        <thead className="bg-gray-50">
                            <tr>
                                <th className="px-4 py-3 text-left font-semibold text-gray-600">ETF</th>
                                {PERFORMANCE_DATA.years.map(year => (
                                    <th key={year} className="px-4 py-3 text-right font-semibold text-gray-600">{year}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-200">
                            {selectedFunds.map(ticker => (
                                <tr key={ticker} className="hover:bg-gray-50">
                                    <td className="px-4 py-3 font-semibold">{ticker}</td>
                                    {PERFORMANCE_DATA[ticker].map((ret, i) => (
                                        <td key={i} className={`px-4 py-3 text-right ${
                                            ret === null ? 'text-gray-400' :
                                            ret >= 0 ? 'text-green-600' : 'text-red-600'
                                        }`}>
                                            {ret === null ? '-' : formatPercent(ret, 1)}
                                        </td>
                                    ))}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};

// Costs Section
const CostsSection = () => {
    const [investment, setInvestment] = useState(10000);
    const [years, setYears] = useState(10);
    const [returnRate, setReturnRate] = useState(7);

    const calculateCost = (expenseRatio) => {
        let balance = investment;
        let totalFees = 0;

        for (let i = 0; i < years; i++) {
            const fee = balance * (expenseRatio / 100);
            totalFees += fee;
            balance = (balance - fee) * (1 + returnRate / 100);
        }

        return { endValue: balance, totalFees };
    };

    const allETFs = [...ETF_DATA.index, ...ETF_DATA.halal, ...ETF_DATA.esg];
    const sortedByExpense = [...allETFs].sort((a, b) => a.expenseRatio - b.expenseRatio);

    const vooResult = calculateCost(0.03);

    return (
        <div className="fade-in space-y-6">
            {/* Cost Calculator */}
            <div className="bg-white rounded-xl p-6 card-shadow">
                <h3 className="font-semibold text-lg mb-4">Fee Impact Calculator</h3>
                <div className="grid md:grid-cols-3 gap-6 mb-6">
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Initial Investment
                        </label>
                        <input
                            type="number"
                            value={investment}
                            onChange={(e) => setInvestment(Number(e.target.value))}
                            className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                        />
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Investment Period (Years)
                        </label>
                        <input
                            type="number"
                            value={years}
                            onChange={(e) => setYears(Number(e.target.value))}
                            className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                        />
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Expected Annual Return (%)
                        </label>
                        <input
                            type="number"
                            value={returnRate}
                            onChange={(e) => setReturnRate(Number(e.target.value))}
                            className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                        />
                    </div>
                </div>

                <div className="overflow-x-auto">
                    <table className="w-full">
                        <thead className="bg-gray-50">
                            <tr>
                                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase">ETF</th>
                                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-600 uppercase">Type</th>
                                <th className="px-4 py-3 text-right text-xs font-semibold text-gray-600 uppercase">Expense Ratio</th>
                                <th className="px-4 py-3 text-right text-xs font-semibold text-gray-600 uppercase">Total Fees Paid</th>
                                <th className="px-4 py-3 text-right text-xs font-semibold text-gray-600 uppercase">End Value</th>
                                <th className="px-4 py-3 text-right text-xs font-semibold text-gray-600 uppercase">Cost vs VOO</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-200">
                            {sortedByExpense.map(etf => {
                                const result = calculateCost(etf.expenseRatio);
                                const costDiff = vooResult.endValue - result.endValue;
                                return (
                                    <tr key={etf.ticker} className="hover:bg-gray-50">
                                        <td className="px-4 py-4 font-semibold text-blue-600">{etf.ticker}</td>
                                        <td className="px-4 py-4">
                                            <span className={`px-2 py-1 rounded-full text-xs font-medium
                                                ${etf.type === 'Index' ? 'bg-gray-100' :
                                                  etf.type === 'Halal' ? 'bg-green-100 text-green-800' :
                                                  'bg-blue-100 text-blue-800'}`}>
                                                {etf.type}
                                            </span>
                                        </td>
                                        <td className="px-4 py-4 text-right">
                                            <span className={etf.expenseRatio > 0.2 ? 'text-red-600 font-medium' : ''}>
                                                {etf.expenseRatio.toFixed(2)}%
                                            </span>
                                        </td>
                                        <td className="px-4 py-4 text-right text-red-600">
                                            ${result.totalFees.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                                        </td>
                                        <td className="px-4 py-4 text-right font-medium">
                                            ${result.endValue.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                                        </td>
                                        <td className="px-4 py-4 text-right">
                                            {costDiff > 0 ? (
                                                <span className="text-red-600">-${costDiff.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
                                            ) : (
                                                <span className="text-green-600">$0</span>
                                            )}
                                        </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Cost Insight Cards */}
            <div className="grid md:grid-cols-2 gap-6">
                <div className="bg-red-50 border border-red-200 rounded-xl p-6">
                    <h4 className="font-semibold text-red-800 mb-3">The Hidden Cost of Values-Based Investing</h4>
                    <ul className="space-y-2 text-sm text-red-700">
                        <li>• SPUS (Halal) charges <strong>16x more</strong> than VOO (0.49% vs 0.03%)</li>
                        <li>• On a $100,000 portfolio over 30 years, this difference costs <strong>over $50,000</strong></li>
                        <li>• ESG funds like ESGU charge 5x more than VOO</li>
                        <li>• Higher fees compound negatively over time</li>
                    </ul>
                </div>
                <div className="bg-blue-50 border border-blue-200 rounded-xl p-6">
                    <h4 className="font-semibold text-blue-800 mb-3">What You're Really Paying For</h4>
                    <ul className="space-y-2 text-sm text-blue-700">
                        <li>• Shariah screening and compliance oversight</li>
                        <li>• Smaller fund sizes = less economies of scale</li>
                        <li>• Active management components</li>
                        <li>• Marketing to niche investor base</li>
                        <li>• Lower competition in specialty fund space</li>
                    </ul>
                </div>
            </div>

            {/* Expense Ratio Comparison Chart */}
            <div className="bg-white rounded-xl p-6 card-shadow">
                <h3 className="font-semibold text-lg mb-4">Expense Ratio Comparison</h3>
                <div className="space-y-4">
                    {sortedByExpense.map(etf => {
                        const maxExpense = 0.65;
                        const width = (etf.expenseRatio / maxExpense) * 100;
                        const colors = {
                            'Index': 'bg-gray-500',
                            'Halal': 'bg-green-500',
                            'ESG': 'bg-blue-500'
                        };
                        return (
                            <div key={etf.ticker} className="flex items-center gap-4">
                                <div className="w-16 font-semibold text-sm">{etf.ticker}</div>
                                <div className="flex-1 h-8 bg-gray-100 rounded-full overflow-hidden">
                                    <div
                                        className={`h-full ${colors[etf.type]} rounded-full flex items-center justify-end pr-3`}
                                        style={{ width: `${Math.max(width, 5)}%` }}
                                    >
                                        <span className="text-white text-xs font-medium">{etf.expenseRatio.toFixed(2)}%</span>
                                    </div>
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>
        </div>
    );
};

// Exclusions Section
const ExclusionsSection = () => {
    const [selectedType, setSelectedType] = useState('halal');

    return (
        <div className="fade-in space-y-6">
            {/* Type Selector */}
            <div className="bg-white rounded-xl p-6 card-shadow">
                <div className="flex gap-4">
                    <button
                        onClick={() => setSelectedType('halal')}
                        className={`px-6 py-3 rounded-lg font-medium transition-colors
                            ${selectedType === 'halal' ? 'bg-green-600 text-white' : 'bg-gray-100 text-gray-600'}`}
                    >
                        Halal/Shariah Exclusions
                    </button>
                    <button
                        onClick={() => setSelectedType('esg')}
                        className={`px-6 py-3 rounded-lg font-medium transition-colors
                            ${selectedType === 'esg' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-600'}`}
                    >
                        ESG Exclusions
                    </button>
                </div>
            </div>

            {/* Exclusion Categories */}
            <div className="bg-white rounded-xl p-6 card-shadow">
                <h3 className="font-semibold text-lg mb-4">
                    {EXCLUSION_CATEGORIES[selectedType].name}
                </h3>
                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {EXCLUSION_CATEGORIES[selectedType].categories.map((cat, i) => (
                        <div key={i} className="border rounded-lg p-4 hover:border-gray-400 transition-colors">
                            <div className="flex items-center justify-between mb-2">
                                <h4 className="font-medium text-gray-800">{cat.name}</h4>
                                <span className={`text-xs px-2 py-1 rounded-full
                                    ${cat.impact === 'High' ? 'bg-red-100 text-red-700' :
                                      cat.impact === 'Medium' ? 'bg-yellow-100 text-yellow-700' :
                                      'bg-green-100 text-green-700'}`}>
                                    {cat.impact} Impact
                                </span>
                            </div>
                            <p className="text-sm text-gray-600 mb-2">{cat.description}</p>
                            <p className="text-sm font-medium text-red-600">
                                ~{cat.pctExcluded}% of market excluded
                            </p>
                        </div>
                    ))}
                </div>
            </div>

            {/* Notable Excluded Companies */}
            <div className="bg-white rounded-xl card-shadow overflow-hidden">
                <div className="p-6 border-b">
                    <h3 className="font-semibold text-lg">Notable Excluded Companies & Their Performance</h3>
                    <p className="text-sm text-gray-500 mt-1">
                        These high-performing stocks are excluded from values-based ETFs
                    </p>
                </div>

                {Object.entries(NOTABLE_EXCLUSIONS).map(([category, stocks]) => (
                    <div key={category} className="border-b last:border-b-0">
                        <div className="px-6 py-3 bg-gray-50">
                            <h4 className="font-medium text-gray-700 capitalize">{category.replace('_', ' ')}</h4>
                        </div>
                        <div className="overflow-x-auto">
                            <table className="w-full text-sm">
                                <thead>
                                    <tr className="text-left text-gray-500">
                                        <th className="px-6 py-2">Ticker</th>
                                        <th className="px-6 py-2">Company</th>
                                        <th className="px-6 py-2 text-right">5Y Return</th>
                                        <th className="px-6 py-2 text-right">Dividend</th>
                                        <th className="px-6 py-2">Excluded By</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {stocks.map(stock => (
                                        <tr key={stock.ticker} className="border-t hover:bg-gray-50">
                                            <td className="px-6 py-3 font-semibold text-blue-600">{stock.ticker}</td>
                                            <td className="px-6 py-3">{stock.name}</td>
                                            <td className="px-6 py-3 text-right">
                                                <span className={stock.return5yr >= 0 ? 'text-green-600' : 'text-red-600'}>
                                                    {formatPercent(stock.return5yr)}
                                                </span>
                                            </td>
                                            <td className="px-6 py-3 text-right">{stock.dividendYield}%</td>
                                            <td className="px-6 py-3">
                                                {stock.excludedBy.map(type => (
                                                    <span key={type} className={`exclusion-badge mr-1
                                                        ${type === 'halal' ? 'bg-green-100 text-green-800' :
                                                          type.includes('esg') ? 'bg-blue-100 text-blue-800' : ''}`}>
                                                        {type}
                                                    </span>
                                                ))}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                ))}
            </div>

            {/* Impact Summary */}
            <div className="grid md:grid-cols-2 gap-6">
                <div className="bg-green-50 border border-green-200 rounded-xl p-6">
                    <h4 className="font-semibold text-green-800 mb-3">Halal ETF Exclusion Impact</h4>
                    <ul className="space-y-2 text-sm text-green-700">
                        <li>• <strong>53% of S&P 500</strong> excluded (~268 companies)</li>
                        <li>• Entire financial sector excluded (JPM, BAC, BRK.B, etc.)</li>
                        <li>• Misses top performers like Visa, Mastercard, Goldman Sachs</li>
                        <li>• High debt screen removes many quality companies</li>
                        <li>• Results in heavy tech concentration (42%+ of portfolio)</li>
                    </ul>
                </div>
                <div className="bg-blue-50 border border-blue-200 rounded-xl p-6">
                    <h4 className="font-semibold text-blue-800 mb-3">ESG ETF Exclusion Impact</h4>
                    <ul className="space-y-2 text-sm text-blue-700">
                        <li>• <strong>15% of market</strong> excluded on average</li>
                        <li>• Energy sector significantly underweight</li>
                        <li>• Missed massive energy rally in 2022 (XOM +80%)</li>
                        <li>• Defense stocks excluded (strong performers)</li>
                        <li>• Subjective ESG ratings vary by provider</li>
                    </ul>
                </div>
            </div>
        </div>
    );
};

// Backtester Section
const BacktesterSection = () => {
    const chartRef = useRef(null);
    const chartInstance = useRef(null);
    const [startAmount, setStartAmount] = useState(10000);
    const [monthlyAdd, setMonthlyAdd] = useState(500);
    const [startDate, setStartDate] = useState('2020-01');
    const [endDate, setEndDate] = useState('2024-12');
    const [selectedFunds, setSelectedFunds] = useState(['SPY', 'SPUS', 'ESGU']);
    const [results, setResults] = useState(null);

    const runBacktest = () => {
        const startIdx = MONTHLY_DATA.months.indexOf(startDate);
        const endIdx = MONTHLY_DATA.months.indexOf(endDate);

        if (startIdx === -1 || endIdx === -1 || startIdx >= endIdx) {
            alert('Invalid date range');
            return;
        }

        const backtestResults = {};

        selectedFunds.forEach(ticker => {
            let balance = startAmount;
            let totalInvested = startAmount;
            const history = [{ month: startDate, value: balance, invested: totalInvested }];

            for (let i = startIdx; i <= endIdx; i++) {
                const monthlyReturn = MONTHLY_DATA[ticker]?.[i] || 0;
                balance = balance * (1 + monthlyReturn / 100);

                if (i < endIdx) {
                    balance += monthlyAdd;
                    totalInvested += monthlyAdd;
                }

                history.push({
                    month: MONTHLY_DATA.months[i],
                    value: balance,
                    invested: totalInvested
                });
            }

            backtestResults[ticker] = {
                finalValue: balance,
                totalInvested,
                totalReturn: ((balance - totalInvested) / totalInvested) * 100,
                profit: balance - totalInvested,
                history
            };
        });

        setResults(backtestResults);
    };

    useEffect(() => {
        if (!results || !chartRef.current) return;

        if (chartInstance.current) {
            chartInstance.current.destroy();
        }

        const ctx = chartRef.current.getContext('2d');

        const colors = {
            SPY: '#374151',
            VOO: '#6b7280',
            SPUS: '#10b981',
            HLAL: '#059669',
            ESGU: '#3b82f6',
            SUSA: '#2563eb',
            ESGV: '#1d4ed8'
        };

        const datasets = Object.entries(results).map(([ticker, data]) => ({
            label: ticker,
            data: data.history.map(h => h.value),
            borderColor: colors[ticker] || '#888',
            backgroundColor: 'transparent',
            borderWidth: 2,
            tension: 0.3,
            pointRadius: 0,
            pointHoverRadius: 4
        }));

        // Add invested line
        const investedData = results[selectedFunds[0]].history.map(h => h.invested);
        datasets.push({
            label: 'Total Invested',
            data: investedData,
            borderColor: '#9ca3af',
            borderDash: [5, 5],
            backgroundColor: 'transparent',
            borderWidth: 1,
            pointRadius: 0
        });

        chartInstance.current = new Chart(ctx, {
            type: 'line',
            data: {
                labels: results[selectedFunds[0]].history.map(h => h.month),
                datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { usePointStyle: true }
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                return `${context.dataset.label}: $${context.parsed.y.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        title: { display: true, text: 'Portfolio Value' },
                        ticks: { callback: (value) => '$' + value.toLocaleString() }
                    },
                    x: {
                        ticks: {
                            maxTicksLimit: 12
                        }
                    }
                }
            }
        });

        return () => {
            if (chartInstance.current) {
                chartInstance.current.destroy();
            }
        };
    }, [results]);

    const allTickers = ['SPY', 'VOO', 'SPUS', 'HLAL', 'ESGU', 'SUSA', 'ESGV'];

    const toggleFund = (ticker) => {
        if (selectedFunds.includes(ticker)) {
            if (selectedFunds.length > 1) {
                setSelectedFunds(selectedFunds.filter(t => t !== ticker));
            }
        } else {
            setSelectedFunds([...selectedFunds, ticker]);
        }
    };

    return (
        <div className="fade-in space-y-6">
            {/* Backtest Configuration */}
            <div className="bg-white rounded-xl p-6 card-shadow">
                <h3 className="font-semibold text-lg mb-4">Portfolio Backtester</h3>

                <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Initial Investment
                        </label>
                        <input
                            type="number"
                            value={startAmount}
                            onChange={(e) => setStartAmount(Number(e.target.value))}
                            className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                        />
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Monthly Contribution
                        </label>
                        <input
                            type="number"
                            value={monthlyAdd}
                            onChange={(e) => setMonthlyAdd(Number(e.target.value))}
                            className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                        />
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Start Date
                        </label>
                        <select
                            value={startDate}
                            onChange={(e) => setStartDate(e.target.value)}
                            className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                        >
                            {MONTHLY_DATA.months.slice(0, -12).map(m => (
                                <option key={m} value={m}>{m}</option>
                            ))}
                        </select>
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            End Date
                        </label>
                        <select
                            value={endDate}
                            onChange={(e) => setEndDate(e.target.value)}
                            className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
                        >
                            {MONTHLY_DATA.months.slice(12).map(m => (
                                <option key={m} value={m}>{m}</option>
                            ))}
                        </select>
                    </div>
                </div>

                <div className="mb-6">
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                        Select ETFs to Compare
                    </label>
                    <div className="flex flex-wrap gap-2">
                        {allTickers.map(ticker => (
                            <button
                                key={ticker}
                                onClick={() => toggleFund(ticker)}
                                className={`px-3 py-1.5 rounded-full text-sm font-medium transition-colors
                                    ${selectedFunds.includes(ticker)
                                        ? 'bg-blue-600 text-white'
                                        : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}
                            >
                                {ticker}
                            </button>
                        ))}
                    </div>
                </div>

                <button
                    onClick={runBacktest}
                    className="px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors"
                >
                    Run Backtest
                </button>
            </div>

            {/* Results */}
            {results && (
                <>
                    {/* Summary Cards */}
                    <div className="grid md:grid-cols-3 lg:grid-cols-4 gap-4">
                        {Object.entries(results).map(([ticker, data]) => {
                            const spyProfit = results.SPY?.profit || data.profit;
                            const diff = data.profit - spyProfit;
                            return (
                                <div key={ticker} className="bg-white rounded-xl p-5 card-shadow">
                                    <h4 className="font-semibold text-lg text-gray-800">{ticker}</h4>
                                    <p className="text-2xl font-bold text-gray-900 mt-2">
                                        ${data.finalValue.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                                    </p>
                                    <p className={`text-sm mt-1 ${data.totalReturn >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                                        {formatPercent(data.totalReturn)} total return
                                    </p>
                                    <p className="text-sm text-gray-500 mt-1">
                                        Invested: ${data.totalInvested.toLocaleString()}
                                    </p>
                                    {ticker !== 'SPY' && results.SPY && (
                                        <p className={`text-sm mt-2 ${diff >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                                            vs SPY: {diff >= 0 ? '+' : ''}${diff.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                                        </p>
                                    )}
                                </div>
                            );
                        })}
                    </div>

                    {/* Chart */}
                    <div className="bg-white rounded-xl p-6 card-shadow">
                        <h3 className="font-semibold text-lg mb-4">Portfolio Growth Over Time</h3>
                        <div className="chart-container">
                            <canvas ref={chartRef}></canvas>
                        </div>
                    </div>

                    {/* Comparison Table */}
                    <div className="bg-white rounded-xl card-shadow overflow-hidden">
                        <div className="p-6 border-b">
                            <h3 className="font-semibold text-lg">Detailed Comparison</h3>
                        </div>
                        <div className="overflow-x-auto">
                            <table className="w-full">
                                <thead className="bg-gray-50">
                                    <tr>
                                        <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase">ETF</th>
                                        <th className="px-6 py-3 text-right text-xs font-semibold text-gray-600 uppercase">Total Invested</th>
                                        <th className="px-6 py-3 text-right text-xs font-semibold text-gray-600 uppercase">Final Value</th>
                                        <th className="px-6 py-3 text-right text-xs font-semibold text-gray-600 uppercase">Profit/Loss</th>
                                        <th className="px-6 py-3 text-right text-xs font-semibold text-gray-600 uppercase">Total Return</th>
                                        <th className="px-6 py-3 text-right text-xs font-semibold text-gray-600 uppercase">Opportunity Cost</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-gray-200">
                                    {Object.entries(results)
                                        .sort((a, b) => b[1].finalValue - a[1].finalValue)
                                        .map(([ticker, data]) => {
                                            const spyValue = results.SPY?.finalValue || data.finalValue;
                                            const opportunityCost = spyValue - data.finalValue;
                                            return (
                                                <tr key={ticker} className="hover:bg-gray-50">
                                                    <td className="px-6 py-4 font-semibold text-blue-600">{ticker}</td>
                                                    <td className="px-6 py-4 text-right">
                                                        ${data.totalInvested.toLocaleString()}
                                                    </td>
                                                    <td className="px-6 py-4 text-right font-medium">
                                                        ${data.finalValue.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                                                    </td>
                                                    <td className={`px-6 py-4 text-right ${data.profit >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                                                        {data.profit >= 0 ? '+' : ''}${data.profit.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                                                    </td>
                                                    <td className={`px-6 py-4 text-right ${data.totalReturn >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                                                        {formatPercent(data.totalReturn)}
                                                    </td>
                                                    <td className="px-6 py-4 text-right">
                                                        {opportunityCost > 0 ? (
                                                            <span className="text-red-600">
                                                                -${opportunityCost.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                                                            </span>
                                                        ) : opportunityCost < 0 ? (
                                                            <span className="text-green-600">
                                                                +${Math.abs(opportunityCost).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                                                            </span>
                                                        ) : (
                                                            <span className="text-gray-400">-</span>
                                                        )}
                                                    </td>
                                                </tr>
                                            );
                                        })}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </>
            )}
        </div>
    );
};

// Holdings Section
const HoldingsSection = () => {
    const [selectedETF, setSelectedETF] = useState('SPY');
    const allETFs = [...ETF_DATA.index, ...ETF_DATA.halal, ...ETF_DATA.esg];
    const etf = allETFs.find(e => e.ticker === selectedETF);

    return (
        <div className="fade-in space-y-6">
            {/* ETF Selector */}
            <div className="bg-white rounded-xl p-6 card-shadow">
                <h3 className="font-semibold text-lg mb-4">Select ETF to Analyze</h3>
                <div className="flex flex-wrap gap-2">
                    {allETFs.map(e => (
                        <button
                            key={e.ticker}
                            onClick={() => setSelectedETF(e.ticker)}
                            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors
                                ${selectedETF === e.ticker
                                    ? 'bg-blue-600 text-white'
                                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}
                        >
                            {e.ticker}
                        </button>
                    ))}
                </div>
            </div>

            {etf && (
                <>
                    {/* ETF Details */}
                    <div className="bg-white rounded-xl p-6 card-shadow">
                        <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-6">
                            <div>
                                <h2 className="text-2xl font-bold text-gray-800">{etf.ticker}</h2>
                                <p className="text-gray-600">{etf.name}</p>
                            </div>
                            <span className={`mt-2 md:mt-0 px-4 py-2 rounded-full text-sm font-medium
                                ${etf.type === 'Index' ? 'bg-gray-100 text-gray-800' :
                                  etf.type === 'Halal' ? 'bg-green-100 text-green-800' :
                                  'bg-blue-100 text-blue-800'}`}>
                                {etf.type}
                            </span>
                        </div>

                        <p className="text-gray-600 mb-6">{etf.description}</p>

                        <div className="grid md:grid-cols-4 gap-4">
                            <StatCard title="Expense Ratio" value={`${etf.expenseRatio.toFixed(2)}%`} color={etf.expenseRatio > 0.2 ? 'red' : 'green'} />
                            <StatCard title="Holdings" value={etf.holdings} color="blue" />
                            <StatCard title="AUM" value={formatCurrency(etf.aum)} color="purple" />
                            <StatCard title="Inception" value={etf.inceptionDate} color="yellow" />
                        </div>
                    </div>

                    {/* Sector Weights */}
                    <div className="bg-white rounded-xl p-6 card-shadow">
                        <h3 className="font-semibold text-lg mb-4">Sector Allocation</h3>
                        <div className="space-y-3">
                            {Object.entries(etf.sectorWeights)
                                .sort((a, b) => b[1] - a[1])
                                .map(([sector, weight]) => (
                                    <div key={sector} className="flex items-center gap-4">
                                        <div className="w-40 text-sm text-gray-600">{sector}</div>
                                        <div className="flex-1 h-6 bg-gray-100 rounded-full overflow-hidden">
                                            <div
                                                className="h-full bg-blue-500 rounded-full"
                                                style={{ width: `${(weight / 45) * 100}%` }}
                                            ></div>
                                        </div>
                                        <div className="w-16 text-right text-sm font-medium">{weight.toFixed(1)}%</div>
                                    </div>
                                ))}
                        </div>
                    </div>

                    {/* Top Holdings */}
                    <div className="bg-white rounded-xl p-6 card-shadow">
                        <h3 className="font-semibold text-lg mb-4">Top 10 Holdings</h3>
                        <div className="grid md:grid-cols-2 lg:grid-cols-5 gap-4">
                            {etf.topHoldings.map((ticker, i) => (
                                <div key={ticker} className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
                                    <span className="text-gray-400 font-medium">{i + 1}</span>
                                    <span className="font-semibold text-blue-600">{ticker}</span>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Exclusions (for Halal/ESG) */}
                    {etf.exclusions && etf.exclusions.length > 0 && (
                        <div className="bg-white rounded-xl p-6 card-shadow">
                            <h3 className="font-semibold text-lg mb-4">Screening Criteria (Exclusions)</h3>
                            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-3">
                                {etf.exclusions.map((exclusion, i) => (
                                    <div key={i} className="flex items-start gap-2 p-3 bg-red-50 rounded-lg">
                                        <span className="text-red-500">✕</span>
                                        <span className="text-sm text-red-800">{exclusion}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Excluded Companies (for Halal/ESG) */}
                    {etf.excludedCompanies && etf.excludedCompanies.length > 0 && (
                        <div className="bg-white rounded-xl p-6 card-shadow">
                            <h3 className="font-semibold text-lg mb-4">Notable Excluded Companies</h3>
                            <div className="flex flex-wrap gap-2">
                                {etf.excludedCompanies.map(ticker => (
                                    <span key={ticker} className="px-3 py-1 bg-gray-100 rounded-full text-sm font-medium text-gray-700">
                                        {ticker}
                                    </span>
                                ))}
                            </div>
                        </div>
                    )}
                </>
            )}
        </div>
    );
};

// Main App Component
const App = () => {
    const [activeTab, setActiveTab] = useState('overview');

    const renderContent = () => {
        switch (activeTab) {
            case 'overview': return <OverviewSection />;
            case 'performance': return <PerformanceSection />;
            case 'costs': return <CostsSection />;
            case 'exclusions': return <ExclusionsSection />;
            case 'backtest': return <BacktesterSection />;
            case 'holdings': return <HoldingsSection />;
            default: return <OverviewSection />;
        }
    };

    return (
        <div className="min-h-screen bg-gray-50">
            <Header />
            <TabNavigation activeTab={activeTab} setActiveTab={setActiveTab} />
            <main className="max-w-7xl mx-auto px-6 py-8">
                {renderContent()}
            </main>
            <footer className="bg-gray-800 text-white py-8 mt-12">
                <div className="max-w-7xl mx-auto px-6 text-center">
                    <p className="text-gray-400 text-sm">
                        Data shown is for illustrative purposes. Past performance does not guarantee future results.
                        Always do your own research before making investment decisions.
                    </p>
                    <p className="text-gray-500 text-xs mt-4">
                        ETF Comparison Tool - Built for educational analysis
                    </p>
                </div>
            </footer>
        </div>
    );
};

// Render the app
ReactDOM.createRoot(document.getElementById('root')).render(<App />);
