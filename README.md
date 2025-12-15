# Aviation MCP Server

[Watch the recording](recording.mp4)


Modified aviation network analysis server with SQL queries and temporary datasets.

All code is in the planes.py folder.
All data needed is in the Air_transportation folder.

## All Functions

### Dataset Management
- `aviation_create_temp_dataset` - Filter data by continent/country/airline/SQL
- `aviation_get_temp_dataset_status` - Check active dataset
- `aviation_clear_temp_dataset` - Reset to global data
- `aviation_get_dataset_schema` - View table structures
- `aviation_sql_query` - Execute custom SQL queries

### Hub & Network Analysis
- `aviation_get_hubs` - Top connected airports
- `aviation_get_transcontinental_hubs` - Airports connecting two continents
- `aviation_find_communities` - Detect flight community clusters

### Route Analysis
- `aviation_find_shortest_path` - Shortest route between airports
- `aviation_nearest_neighbors` - Find nearest airports/navaids

### Geographic Analysis
- `aviation_find_outliers` - Geographic extremes (north/south/east/west/highest)

### Airline & Country Stats
- `aviation_analyze_countries` - Country connectivity metrics
- `aviation_analyze_airline` - Airline route network analysis
- `aviation_analyze_equipment` - Aircraft usage statistics

### Airport Info
- `aviation_get_airport_info` - Detailed airport information

## Quick Example

```python
# Filter to Europe
aviation_create_temp_dataset(filter_type="continent", filter_value="EU")

# All tools now use EU data automatically
aviation_get_hubs(top_n=10)
aviation_find_shortest_path(start_airport="LIS", end_airport="ATH")

# Clear filter
aviation_clear_temp_dataset()
```

## Data Files Needed

Place in `Air_transportation/` directory:
- airports_new.csv, airline_routes.json, routes_new.dat
- navaids_new.csv, airlines_new.dat, countries_20190227.csv
