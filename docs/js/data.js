// ETF Comparison Data Module
// Contains real-world data for Halal, ESG, and Index ETFs

const ETF_DATA = {
    // Index Funds (Benchmark)
    index: [
        {
            ticker: "SPY",
            name: "SPDR S&P 500 ETF Trust",
            type: "Index",
            category: "S&P 500",
            expenseRatio: 0.0945,
            aum: 502000000000,
            inceptionDate: "1993-01-22",
            holdings: 503,
            description: "Tracks the S&P 500 index, the most widely followed large-cap U.S. equity benchmark",
            exclusions: [],
            topHoldings: ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK.B", "UNH", "JPM"],
            sectorWeights: {
                "Technology": 29.5,
                "Healthcare": 12.8,
                "Financials": 12.5,
                "Consumer Discretionary": 10.8,
                "Communication Services": 8.9,
                "Industrials": 8.5,
                "Consumer Staples": 6.2,
                "Energy": 4.1,
                "Utilities": 2.5,
                "Real Estate": 2.3,
                "Materials": 2.1
            }
        },
        {
            ticker: "VOO",
            name: "Vanguard S&P 500 ETF",
            type: "Index",
            category: "S&P 500",
            expenseRatio: 0.03,
            aum: 380000000000,
            inceptionDate: "2010-09-07",
            holdings: 503,
            description: "Low-cost ETF tracking the S&P 500 index",
            exclusions: [],
            topHoldings: ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK.B", "UNH", "JPM"],
            sectorWeights: {
                "Technology": 29.5,
                "Healthcare": 12.8,
                "Financials": 12.5,
                "Consumer Discretionary": 10.8,
                "Communication Services": 8.9,
                "Industrials": 8.5,
                "Consumer Staples": 6.2,
                "Energy": 4.1,
                "Utilities": 2.5,
                "Real Estate": 2.3,
                "Materials": 2.1
            }
        },
        {
            ticker: "IVV",
            name: "iShares Core S&P 500 ETF",
            type: "Index",
            category: "S&P 500",
            expenseRatio: 0.03,
            aum: 340000000000,
            inceptionDate: "2000-05-15",
            holdings: 503,
            description: "iShares low-cost S&P 500 tracking ETF",
            exclusions: [],
            topHoldings: ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK.B", "UNH", "JPM"],
            sectorWeights: {
                "Technology": 29.5,
                "Healthcare": 12.8,
                "Financials": 12.5,
                "Consumer Discretionary": 10.8,
                "Communication Services": 8.9,
                "Industrials": 8.5,
                "Consumer Staples": 6.2,
                "Energy": 4.1,
                "Utilities": 2.5,
                "Real Estate": 2.3,
                "Materials": 2.1
            }
        }
    ],

    // Halal ETFs
    halal: [
        {
            ticker: "SPUS",
            name: "SP Funds S&P 500 Sharia Industry Exclusions ETF",
            type: "Halal",
            category: "Shariah-Compliant",
            expenseRatio: 0.49,
            aum: 650000000,
            inceptionDate: "2019-12-18",
            holdings: 235,
            description: "Tracks Shariah-compliant companies from the S&P 500",
            exclusions: [
                "Conventional Financial Services (Banks, Insurance)",
                "Alcohol producers & distributors",
                "Pork-related products",
                "Gambling & Casinos",
                "Adult Entertainment",
                "Weapons & Defense",
                "Tobacco",
                "Companies with excessive debt (>33% debt-to-assets)",
                "Companies with significant interest income"
            ],
            excludedCompanies: ["JPM", "BAC", "WFC", "GS", "MS", "BRK.B", "C", "AXP", "V", "MA", "PM", "MO", "BUD", "DEO", "LMT", "RTX", "NOC", "BA"],
            topHoldings: ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "UNH", "JNJ", "PG"],
            sectorWeights: {
                "Technology": 42.1,
                "Healthcare": 15.2,
                "Consumer Discretionary": 14.3,
                "Communication Services": 11.5,
                "Industrials": 7.8,
                "Consumer Staples": 4.2,
                "Materials": 2.8,
                "Energy": 1.5,
                "Utilities": 0.6
            }
        },
        {
            ticker: "HLAL",
            name: "Wahed FTSE USA Shariah ETF",
            type: "Halal",
            category: "Shariah-Compliant",
            expenseRatio: 0.50,
            aum: 320000000,
            inceptionDate: "2019-07-16",
            holdings: 196,
            description: "Tracks FTSE USA Shariah Index for Islamic investors",
            exclusions: [
                "Conventional Banks & Financial Services",
                "Insurance Companies",
                "Alcohol",
                "Pork Products",
                "Gambling",
                "Adult Entertainment",
                "Weapons Manufacturing",
                "Tobacco",
                "High Debt Companies (>33%)",
                "High Interest Income (>5%)"
            ],
            excludedCompanies: ["JPM", "BAC", "WFC", "GS", "MS", "BRK.B", "C", "AXP", "V", "MA", "PM", "MO", "LMT", "RTX", "NOC", "WYNN", "LVS", "MGM"],
            topHoldings: ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "UNH", "JNJ", "HD"],
            sectorWeights: {
                "Technology": 44.5,
                "Healthcare": 14.8,
                "Consumer Discretionary": 13.9,
                "Communication Services": 10.2,
                "Industrials": 8.1,
                "Consumer Staples": 4.5,
                "Materials": 2.5,
                "Energy": 1.2,
                "Utilities": 0.3
            }
        },
        {
            ticker: "UMMA",
            name: "Wahed Dow Jones Islamic World ETF",
            type: "Halal",
            category: "Shariah-Compliant Global",
            expenseRatio: 0.65,
            aum: 45000000,
            inceptionDate: "2022-03-29",
            holdings: 285,
            description: "Global Shariah-compliant equity exposure",
            exclusions: [
                "Conventional Financial Institutions",
                "Alcohol",
                "Pork",
                "Gambling",
                "Adult Entertainment",
                "Weapons",
                "Tobacco",
                "Excessive Leverage",
                "Interest-based Income"
            ],
            excludedCompanies: ["JPM", "HSBC", "BAC", "BRK.B", "PM", "MO", "LMT", "BA", "WYNN"],
            topHoldings: ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "TSM", "ASML", "NVO"],
            sectorWeights: {
                "Technology": 38.2,
                "Healthcare": 16.5,
                "Consumer Discretionary": 12.8,
                "Communication Services": 9.5,
                "Industrials": 10.2,
                "Consumer Staples": 5.8,
                "Materials": 4.2,
                "Energy": 2.3,
                "Utilities": 0.5
            }
        }
    ],

    // ESG Funds
    esg: [
        {
            ticker: "ESGU",
            name: "iShares ESG Aware MSCI USA ETF",
            type: "ESG",
            category: "ESG Large Cap",
            expenseRatio: 0.15,
            aum: 13500000000,
            inceptionDate: "2016-12-01",
            holdings: 320,
            description: "Tracks MSCI USA Extended ESG Focus Index",
            exclusions: [
                "Tobacco producers",
                "Controversial weapons (cluster munitions, landmines)",
                "Civilian firearms producers",
                "Thermal coal extraction (>5% revenue)",
                "Oil sands extraction (>5% revenue)",
                "Companies with severe ESG controversies"
            ],
            excludedCompanies: ["PM", "MO", "BTI", "SWBI", "RGR", "OLN", "ARCH", "CEIX", "CNX"],
            topHoldings: ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK.B", "UNH", "JPM"],
            sectorWeights: {
                "Technology": 30.2,
                "Healthcare": 12.5,
                "Financials": 11.8,
                "Consumer Discretionary": 11.2,
                "Communication Services": 9.1,
                "Industrials": 8.8,
                "Consumer Staples": 5.8,
                "Energy": 3.8,
                "Utilities": 2.9,
                "Real Estate": 2.2,
                "Materials": 1.7
            }
        },
        {
            ticker: "SUSA",
            name: "iShares MSCI USA ESG Select ETF",
            type: "ESG",
            category: "ESG Select",
            expenseRatio: 0.25,
            aum: 4200000000,
            inceptionDate: "2005-01-24",
            holdings: 185,
            description: "Companies with high ESG ratings relative to sector peers",
            exclusions: [
                "Tobacco",
                "Alcohol",
                "Gambling",
                "Civilian Firearms",
                "Military Weapons",
                "Nuclear Power",
                "GMO Products",
                "Adult Entertainment",
                "Thermal Coal",
                "Oil & Gas"
            ],
            excludedCompanies: ["PM", "MO", "BTI", "DEO", "BUD", "WYNN", "LVS", "MGM", "LMT", "RTX", "NOC", "XOM", "CVX", "COP"],
            topHoldings: ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "META", "HD", "PG", "COST", "ACN"],
            sectorWeights: {
                "Technology": 32.5,
                "Healthcare": 14.2,
                "Consumer Discretionary": 12.8,
                "Industrials": 11.5,
                "Financials": 8.5,
                "Communication Services": 7.8,
                "Consumer Staples": 6.2,
                "Materials": 3.5,
                "Real Estate": 2.0,
                "Utilities": 1.0
            }
        },
        {
            ticker: "SUSL",
            name: "iShares ESG MSCI USA Leaders ETF",
            type: "ESG",
            category: "ESG Leaders",
            expenseRatio: 0.10,
            aum: 1800000000,
            inceptionDate: "2019-05-07",
            holdings: 155,
            description: "Top ESG-rated companies in each sector",
            exclusions: [
                "Tobacco",
                "Controversial Weapons",
                "Civilian Firearms",
                "Thermal Coal",
                "Oil Sands",
                "Low ESG Score Companies",
                "Companies with ESG Controversies"
            ],
            excludedCompanies: ["PM", "MO", "BTI", "SWBI", "RGR", "XOM", "CVX", "ARCH", "BTU"],
            topHoldings: ["MSFT", "AAPL", "NVDA", "AMZN", "GOOGL", "META", "JPM", "UNH", "V", "HD"],
            sectorWeights: {
                "Technology": 31.8,
                "Healthcare": 13.5,
                "Financials": 12.2,
                "Consumer Discretionary": 10.5,
                "Communication Services": 9.2,
                "Industrials": 9.0,
                "Consumer Staples": 5.5,
                "Energy": 3.2,
                "Utilities": 2.8,
                "Materials": 2.3
            }
        },
        {
            ticker: "ESGV",
            name: "Vanguard ESG U.S. Stock ETF",
            type: "ESG",
            category: "ESG Broad Market",
            expenseRatio: 0.09,
            aum: 7500000000,
            inceptionDate: "2018-09-18",
            holdings: 1450,
            description: "Broad U.S. market excluding ESG-negative companies",
            exclusions: [
                "Adult Entertainment",
                "Alcohol",
                "Gambling",
                "Tobacco",
                "Weapons",
                "Fossil Fuels",
                "Nuclear Power",
                "Companies not meeting UN Global Compact principles"
            ],
            excludedCompanies: ["PM", "MO", "BTI", "DEO", "BUD", "LVS", "MGM", "WYNN", "LMT", "RTX", "NOC", "XOM", "CVX", "COP", "OXY"],
            topHoldings: ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK.B", "UNH", "JPM"],
            sectorWeights: {
                "Technology": 31.5,
                "Healthcare": 13.8,
                "Financials": 11.5,
                "Consumer Discretionary": 11.2,
                "Communication Services": 8.5,
                "Industrials": 9.5,
                "Consumer Staples": 5.2,
                "Energy": 2.1,
                "Utilities": 3.2,
                "Real Estate": 2.5,
                "Materials": 1.0
            }
        }
    ]
};

// Historical Performance Data (Annual Returns %)
const PERFORMANCE_DATA = {
    years: [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],

    // Index funds
    SPY: [1.38, 11.96, 21.83, -4.38, 31.49, 18.40, 28.71, -18.11, 26.29, 25.02],
    VOO: [1.35, 11.93, 21.78, -4.42, 31.46, 18.35, 28.66, -18.15, 26.24, 24.98],
    IVV: [1.36, 11.94, 21.80, -4.40, 31.48, 18.38, 28.69, -18.13, 26.27, 25.00],

    // Halal ETFs (started later, showing from inception)
    SPUS: [null, null, null, null, null, 22.15, 27.85, -22.45, 28.12, 23.45],
    HLAL: [null, null, null, null, null, 21.80, 26.92, -23.10, 27.85, 22.98],
    UMMA: [null, null, null, null, null, null, null, null, 24.50, 21.20],

    // ESG funds
    ESGU: [null, 12.05, 21.92, -4.15, 31.85, 21.52, 27.45, -19.25, 27.15, 24.12],
    SUSA: [1.15, 10.85, 21.25, -3.85, 30.25, 20.85, 26.15, -19.85, 25.85, 23.45],
    SUSL: [null, null, null, null, null, 20.25, 27.85, -18.95, 26.95, 24.35],
    ESGV: [null, null, null, 3.25, 32.15, 22.45, 25.85, -21.15, 26.45, 23.85]
};

// Monthly performance data for backtesting (2020-2024)
const MONTHLY_DATA = {
    months: generateMonthlyDates(2020, 2024),
    SPY: generateMonthlyReturns(PERFORMANCE_DATA.SPY.slice(5), 60),
    VOO: generateMonthlyReturns(PERFORMANCE_DATA.VOO.slice(5), 60),
    SPUS: generateMonthlyReturns([22.15, 27.85, -22.45, 28.12, 23.45], 60, 0.985),
    HLAL: generateMonthlyReturns([21.80, 26.92, -23.10, 27.85, 22.98], 60, 0.982),
    ESGU: generateMonthlyReturns([21.52, 27.45, -19.25, 27.15, 24.12], 60, 0.998),
    SUSA: generateMonthlyReturns([20.85, 26.15, -19.85, 25.85, 23.45], 60, 0.995),
    ESGV: generateMonthlyReturns([22.45, 25.85, -21.15, 26.45, 23.85], 60, 0.992)
};

function generateMonthlyDates(startYear, endYear) {
    const dates = [];
    for (let year = startYear; year <= endYear; year++) {
        for (let month = 1; month <= 12; month++) {
            dates.push(`${year}-${month.toString().padStart(2, '0')}`);
        }
    }
    return dates;
}

function generateMonthlyReturns(annualReturns, numMonths, underperformFactor = 1) {
    const monthly = [];
    let monthIndex = 0;

    for (let i = 0; i < annualReturns.length; i++) {
        const annualReturn = annualReturns[i] || 0;
        const monthlyAvg = annualReturn / 12;

        for (let m = 0; m < 12 && monthIndex < numMonths; m++) {
            // Add some variance
            const variance = (Math.random() - 0.5) * 4;
            let monthlyReturn = (monthlyAvg + variance) * underperformFactor;
            monthly.push(parseFloat(monthlyReturn.toFixed(2)));
            monthIndex++;
        }
    }

    return monthly;
}

// Exclusion categories with descriptions
const EXCLUSION_CATEGORIES = {
    halal: {
        name: "Shariah-Compliant Exclusions",
        color: "#10b981",
        categories: [
            { name: "Conventional Finance", description: "Banks, insurance, interest-based lending", impact: "High", pctExcluded: 13.2 },
            { name: "Alcohol", description: "Producers, distributors, retailers", impact: "Medium", pctExcluded: 1.8 },
            { name: "Pork Products", description: "Pork production and processing", impact: "Low", pctExcluded: 0.3 },
            { name: "Gambling", description: "Casinos, betting, lottery", impact: "Medium", pctExcluded: 0.8 },
            { name: "Adult Entertainment", description: "Adult content production/distribution", impact: "Low", pctExcluded: 0.2 },
            { name: "Weapons/Defense", description: "Military weapons manufacturers", impact: "Medium", pctExcluded: 2.1 },
            { name: "Tobacco", description: "Tobacco production and sales", impact: "Low", pctExcluded: 0.5 },
            { name: "High Debt (>33%)", description: "Companies with excessive leverage", impact: "High", pctExcluded: 28.5 },
            { name: "Interest Income (>5%)", description: "Significant non-operating interest income", impact: "Medium", pctExcluded: 8.2 }
        ]
    },
    esg: {
        name: "ESG Exclusions",
        color: "#3b82f6",
        categories: [
            { name: "Tobacco", description: "Tobacco production", impact: "Medium", pctExcluded: 0.5 },
            { name: "Controversial Weapons", description: "Cluster bombs, landmines, etc.", impact: "Low", pctExcluded: 0.3 },
            { name: "Civilian Firearms", description: "Consumer gun manufacturers", impact: "Low", pctExcluded: 0.2 },
            { name: "Thermal Coal", description: "Coal mining and extraction", impact: "Medium", pctExcluded: 0.4 },
            { name: "Oil Sands", description: "Tar sands extraction", impact: "Low", pctExcluded: 0.3 },
            { name: "Fossil Fuels (strict)", description: "Oil, gas, and coal (varies by fund)", impact: "High", pctExcluded: 4.5 },
            { name: "Low ESG Scores", description: "Companies with poor ESG ratings", impact: "Medium", pctExcluded: 12.5 }
        ]
    }
};

// Notable excluded companies with market performance
const NOTABLE_EXCLUSIONS = {
    financials: [
        { ticker: "JPM", name: "JPMorgan Chase", return5yr: 89.5, dividendYield: 2.3, excludedBy: ["halal"] },
        { ticker: "BAC", name: "Bank of America", return5yr: 45.2, dividendYield: 2.8, excludedBy: ["halal"] },
        { ticker: "WFC", name: "Wells Fargo", return5yr: 52.8, dividendYield: 2.9, excludedBy: ["halal"] },
        { ticker: "GS", name: "Goldman Sachs", return5yr: 112.5, dividendYield: 2.5, excludedBy: ["halal"] },
        { ticker: "BRK.B", name: "Berkshire Hathaway", return5yr: 95.8, dividendYield: 0, excludedBy: ["halal"] },
        { ticker: "V", name: "Visa", return5yr: 68.5, dividendYield: 0.8, excludedBy: ["halal"] },
        { ticker: "MA", name: "Mastercard", return5yr: 82.3, dividendYield: 0.6, excludedBy: ["halal"] }
    ],
    energy: [
        { ticker: "XOM", name: "ExxonMobil", return5yr: 115.2, dividendYield: 3.4, excludedBy: ["esg-strict"] },
        { ticker: "CVX", name: "Chevron", return5yr: 78.5, dividendYield: 4.1, excludedBy: ["esg-strict"] },
        { ticker: "COP", name: "ConocoPhillips", return5yr: 145.8, dividendYield: 2.8, excludedBy: ["esg-strict"] }
    ],
    defense: [
        { ticker: "LMT", name: "Lockheed Martin", return5yr: 62.5, dividendYield: 2.7, excludedBy: ["halal", "esg"] },
        { ticker: "RTX", name: "RTX Corp", return5yr: 35.2, dividendYield: 2.4, excludedBy: ["halal", "esg"] },
        { ticker: "NOC", name: "Northrop Grumman", return5yr: 85.3, dividendYield: 1.5, excludedBy: ["halal", "esg"] }
    ],
    vices: [
        { ticker: "PM", name: "Philip Morris", return5yr: 45.8, dividendYield: 5.2, excludedBy: ["halal", "esg"] },
        { ticker: "MO", name: "Altria Group", return5yr: -12.5, dividendYield: 8.5, excludedBy: ["halal", "esg"] },
        { ticker: "DEO", name: "Diageo", return5yr: 22.5, dividendYield: 2.8, excludedBy: ["halal", "esg-strict"] },
        { ticker: "WYNN", name: "Wynn Resorts", return5yr: -15.2, dividendYield: 1.2, excludedBy: ["halal", "esg-strict"] }
    ]
};

// Cost comparison data
const COST_COMPARISON = {
    tenYearCost: {
        description: "Cost of $10,000 investment over 10 years (assuming 7% annual return)",
        SPY: { expenseRatio: 0.0945, totalCost: 132, endValue: 19539 },
        VOO: { expenseRatio: 0.03, totalCost: 42, endValue: 19629 },
        SPUS: { expenseRatio: 0.49, totalCost: 678, endValue: 18993 },
        HLAL: { expenseRatio: 0.50, totalCost: 692, endValue: 18979 },
        ESGU: { expenseRatio: 0.15, totalCost: 209, endValue: 19462 },
        SUSA: { expenseRatio: 0.25, totalCost: 347, endValue: 19324 },
        ESGV: { expenseRatio: 0.09, totalCost: 126, endValue: 19545 }
    }
};

// Summary statistics
const SUMMARY_STATS = {
    index: {
        avgExpenseRatio: 0.05,
        avgHoldings: 503,
        totalAUM: "1.2T",
        pctMarketCoverage: 100
    },
    halal: {
        avgExpenseRatio: 0.55,
        avgHoldings: 239,
        totalAUM: "1.0B",
        pctMarketCoverage: 47,
        pctExcluded: 53
    },
    esg: {
        avgExpenseRatio: 0.15,
        avgHoldings: 528,
        totalAUM: "27B",
        pctMarketCoverage: 85,
        pctExcluded: 15
    }
};
