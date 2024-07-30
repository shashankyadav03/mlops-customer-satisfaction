import urllib.parse
import webbrowser

# List of companies
companies = [
    "Monzo"
]

# Base LinkedIn search URL
base_url = "https://www.linkedin.com/search/results/people/?keywords="

# Generate LinkedIn search URLs
linkedin_urls = [base_url + urllib.parse.quote(company) for company in companies]

# Display the URLs
for url in linkedin_urls:
    print(url)
    # open url to open to open in     
    webbrowser.open(url)
