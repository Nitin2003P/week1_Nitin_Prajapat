# Configuration file for Country Intelligence System

# App Configuration
APP_TITLE = "🌍 Country Intelligence System"
APP_SUBTITLE = "AI-Powered Aid Distribution for HELP International"
APP_ICON = "🌍"

# Model Parameters
OPTIMAL_CLUSTERS = 3
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Feature Descriptions
FEATURE_DESCRIPTIONS = {
    'country': 'Country name',
    'child_mort': 'Death of children under 5 years of age per 1000 live births',
    'exports': 'Exports of goods and services per capita',
    'health': 'Total health spending as % of GDP',
    'imports': 'Imports of goods and services per capita',
    'income': 'Net income per person',
    'inflation': 'The measurement of the annual growth rate of the Total GDP',
    'life_expec': 'The average number of years a child would live if current mortality patterns continue',
    'total_fer': 'The number of children that would be born to each woman',
    'gdpp': 'The GDP per capita'
}

# Priority Weights for calculating priority score
PRIORITY_WEIGHTS = {
    'child_mort': 0.3,
    'life_expec': 0.25,
    'income': 0.2,
    'total_fer': 0.15,
    'gdpp': 0.1
}