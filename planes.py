"""
Aviation Network Analysis MCP Server

An MCP server that provides tools for analyzing global aviation networks,
including airport hubs, route analysis, flight communities, geographic outliers,
nearest neighbors, and equipment/airline analysis.

This version includes SQL query capabilities for flexible dataset filtering
and has removed all map and image generation features.

Requires data files:
- airports_new.csv
- airline_routes.json
- routes_new.dat
- navaids_new.csv
- airlines_new.dat
- countries_20190227.csv
"""

import json
import os
import sqlite3
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from pathlib import Path

import pandas as pd
import numpy as np
import networkx as nx
import networkx.algorithms.community as nx_comm
from sklearn.neighbors import BallTree
from sklearn.cluster import KMeans
from pydantic import BaseModel, Field, ConfigDict

from mcp.server.fastmcp import FastMCP

# --- Configuration ---
EARTH_RADIUS_KM = 6371
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Air_transportation")

# Initialize the MCP server
mcp = FastMCP("planes")

# --- Global Data (loaded on startup) ---
_data_loaded = False
airports: pd.DataFrame = None
routes_json: dict = None
routes_dat: pd.DataFrame = None
routes_merged: pd.DataFrame = None
navaids: pd.DataFrame = None
airlines: pd.DataFrame = None
countries: pd.DataFrame = None
G_global: nx.DiGraph = None
db_conn: sqlite3.Connection = None

# --- Temporary/Filtered Dataset ---
temp_dataset_active = False
temp_airports: pd.DataFrame = None
temp_routes_dat: pd.DataFrame = None
temp_routes_merged: pd.DataFrame = None
temp_G: nx.DiGraph = None
temp_filter_description: str = None


def _ensure_data_loaded():
    """Ensures all data is loaded before any tool runs."""
    global _data_loaded, airports, routes_json, routes_dat, routes_merged
    global navaids, airlines, countries, G_global, db_conn
    
    if _data_loaded:
        return
    
    # Load Airports
    airports = pd.read_csv(
        os.path.join(DATA_DIR, 'airports_new.csv'), 
        keep_default_na=False, 
        na_values=['']
    )
    airports = airports.dropna(subset=['iata_code']).drop_duplicates(subset=['iata_code'])
    airports.set_index('iata_code', inplace=True)
    
    # Load Routes (JSON)
    with open(os.path.join(DATA_DIR, 'airline_routes.json'), 'r') as f:
        routes_json = json.load(f)
    
    # Load Routes (DAT)
    route_cols = ['Airline', 'AirlineID', 'Source', 'SourceID', 'Dest', 'DestID', 'Codeshare', 'Stops', 'Equipment']
    routes_dat = pd.read_csv(
        os.path.join(DATA_DIR, 'routes_new.dat'), 
        header=None, 
        names=route_cols
    )
    routes_dat['Source'] = routes_dat['Source'].astype(str)
    routes_dat['Dest'] = routes_dat['Dest'].astype(str)
    
    # Load Navaids
    navaids = pd.read_csv(os.path.join(DATA_DIR, 'navaids_new.csv'))
    navaids = navaids.dropna(subset=['latitude_deg', 'longitude_deg'])
    
    # Load Airlines
    airlines = pd.read_csv(
        os.path.join(DATA_DIR, 'airlines_new.dat'), 
        header=None,
        names=['AirlineID', 'Name', 'Alias', 'IATA', 'ICAO', 'Callsign', 'Country', 'Active']
    )
    airlines.set_index('IATA', inplace=True)
    
    # Load Countries
    try:
        countries = pd.read_csv(
            os.path.join(DATA_DIR, 'countries_20190227.csv'), 
            usecols=['code', 'name']
        )
        countries.set_index('code', inplace=True)
    except FileNotFoundError:
        countries = pd.DataFrame()
    
    # Build Network Graph
    G_global = nx.DiGraph()
    valid_nodes = set(airports.index)
    for source, attributes in routes_json.items():
        if source in valid_nodes:
            for route in attributes.get('routes', []):
                target = route.get('iata')
                if target in valid_nodes:
                    G_global.add_edge(source, target, weight=route.get('km', 0))
    
    # Pre-merge Route Data with Geo Info
    routes_merged = routes_dat.merge(
        airports[['iso_country', 'continent']], 
        left_on='Source', 
        right_index=True,
        how='inner'
    )
    routes_merged = routes_merged.rename(columns={'iso_country': 'src_country', 'continent': 'src_continent'})
    
    routes_merged = routes_merged.merge(
        airports[['iso_country', 'continent']], 
        left_on='Dest', 
        right_index=True,
        how='inner'
    )
    routes_merged = routes_merged.rename(columns={'iso_country': 'dst_country', 'continent': 'dst_continent'})
    
    # Initialize in-memory SQLite database for SQL queries
    db_conn = sqlite3.connect(':memory:')
    
    # Load all dataframes into SQLite
    airports_reset = airports.reset_index()
    airports_reset.to_sql('airports', db_conn, index=False, if_exists='replace')
    routes_dat.to_sql('routes', db_conn, index=False, if_exists='replace')
    routes_merged.to_sql('routes_merged', db_conn, index=False, if_exists='replace')
    navaids.to_sql('navaids', db_conn, index=False, if_exists='replace')
    airlines_reset = airlines.reset_index()
    airlines_reset.to_sql('airlines', db_conn, index=False, if_exists='replace')
    if not countries.empty:
        countries_reset = countries.reset_index()
        countries_reset.to_sql('countries', db_conn, index=False, if_exists='replace')
    
    _data_loaded = True


def _get_active_datasets():
    """Returns the currently active datasets (temp if active, otherwise global)."""
    global temp_dataset_active, temp_airports, temp_routes_dat, temp_routes_merged, temp_G
    global airports, routes_dat, routes_merged, G_global
    
    if temp_dataset_active:
        return {
            'airports': temp_airports,
            'routes_dat': temp_routes_dat,
            'routes_merged': temp_routes_merged,
            'graph': temp_G,
            'is_temp': True,
            'description': temp_filter_description
        }
    else:
        return {
            'airports': airports,
            'routes_dat': routes_dat,
            'routes_merged': routes_merged,
            'graph': G_global,
            'is_temp': False,
            'description': 'Full global dataset'
        }


def _create_graph_from_routes(airports_df, routes_data):
    """Helper function to create a NetworkX graph from routes data."""
    G = nx.DiGraph()
    valid_nodes = set(airports_df.index)
    
    # Build from routes_json structure if available
    for source in routes_data['Source'].unique():
        if source in valid_nodes:
            dest_routes = routes_data[routes_data['Source'] == source]
            for _, route in dest_routes.iterrows():
                target = route['Dest']
                if target in valid_nodes:
                    # Try to get distance from global routes_json if available
                    distance = 0
                    if source in routes_json and routes_json[source].get('routes'):
                        for r in routes_json[source]['routes']:
                            if r.get('iata') == target:
                                distance = r.get('km', 0)
                                break
                    G.add_edge(source, target, weight=distance)
    
    return G


# --- Response Format Enum ---
class ResponseFormat(str, Enum):
    """Output format for tool responses."""
    MARKDOWN = "markdown"
    JSON = "json"
    CSV = "csv"


# =============================================================================
# Input Models
# =============================================================================

class GetHubsInput(BaseModel):
    """Input for finding top hub airports."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    top_n: int = Field(
        default=10, 
        description="Number of top hubs to return (1-50)", 
        ge=1, le=50
    )
    continent_filter: Optional[str] = Field(
        default=None, 
        description="Filter to specific continent code (e.g., 'EU', 'NA', 'AS', 'AF', 'SA', 'OC')"
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN, 
        description="Output format: 'markdown', 'json', or 'csv'"
    )


class GetTranscontinentalHubsInput(BaseModel):
    """Input for finding transcontinental hub airports."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    continent_a: str = Field(
        ..., 
        description="First continent code (e.g., 'NA' for North America)"
    )
    continent_b: str = Field(
        ..., 
        description="Second continent code (e.g., 'EU' for Europe)"
    )
    top_n: int = Field(default=10, description="Number of top hubs to return", ge=1, le=50)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class CountryAnalysisInput(BaseModel):
    """Input for country connectivity analysis."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    top_n: int = Field(default=10, description="Number of top countries to show", ge=1, le=50)
    metric: str = Field(
        default="total", 
        description="Metric to sort by: 'total', 'domestic', 'international', 'intercontinental', "
                    "'domestic_rate', 'international_rate', 'intercontinental_rate'"
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class FlightCommunitiesInput(BaseModel):
    """Input for analyzing flight communities."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    top_n: int = Field(default=5, description="Number of top communities to detail", ge=1, le=20)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class OutliersInput(BaseModel):
    """Input for finding geographic outliers."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    region_filter: str = Field(
        ..., 
        description="Region code to analyze (e.g., 'EU' for Europe, 'DE' for Germany)"
    )
    region_name: str = Field(
        ..., 
        description="Human-readable region name for display (e.g., 'Europe', 'Germany')"
    )
    filter_type: str = Field(
        default="continent", 
        description="Type of filter: 'continent' or 'country'"
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class NearestNeighborsInput(BaseModel):
    """Input for finding nearest airports and navaids."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    query: str = Field(
        ..., 
        description="Airport IATA code (e.g., 'LHR') or coordinates as 'lat,lon' (e.g., '51.47,-0.45')"
    )
    k: int = Field(default=5, description="Number of nearest items to find", ge=1, le=20)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class AirlineAnalysisInput(BaseModel):
    """Input for airline operational analysis."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    airline_code: str = Field(
        ..., 
        description="Airline IATA code (e.g., 'LH' for Lufthansa, 'UA' for United)"
    )
    top_n: int = Field(
        default=5, 
        description="Number of top destinations/equipment to show", 
        ge=1, le=20
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class EquipmentAnalysisInput(BaseModel):
    """Input for aircraft equipment analysis."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    top_n: int = Field(default=15, description="Number of top aircraft types to show", ge=1, le=50)
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class AirportInfoInput(BaseModel):
    """Input for getting airport information."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    airport_code: str = Field(
        ..., 
        description="Airport IATA code (e.g., 'JFK', 'LHR', 'NRT')"
    )
    include_routes: bool = Field(
        default=True, 
        description="Include route information"
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class KMeansClusterInput(BaseModel):
    """Input for K-Means clustering of airports."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    k: int = Field(default=8, description="Number of clusters", ge=2, le=20)
    continent: Optional[str] = Field(
        default=None, 
        description="Filter to specific continent (e.g., 'EU', 'NA')"
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN)


class SQLQueryInput(BaseModel):
    """Input for executing SQL queries on aviation datasets."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    query: str = Field(
        ...,
        description="SQL query to execute. Available tables: airports, routes, routes_merged, navaids, airlines, countries"
    )
    limit: Optional[int] = Field(
        default=100,
        description="Maximum number of rows to return (default: 100, max: 10000)",
        ge=1,
        le=10000
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown', 'json', or 'csv'"
    )


class ShortestPathInput(BaseModel):
    """Input for finding shortest flight path between airports."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    start_airport: str = Field(
        ...,
        description="Starting airport IATA code (e.g., 'JFK', 'EZE')"
    )
    end_airport: str = Field(
        ...,
        description="Destination airport IATA code (e.g., 'SYD', 'HND')"
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown', 'json', or 'csv'"
    )


class CreateTempDatasetInput(BaseModel):
    """Input for creating a filtered temporary dataset."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    filter_type: str = Field(
        ...,
        description="Type of filter: 'continent', 'country', 'airline', 'sql' or 'custom'"
    )
    filter_value: str = Field(
        ...,
        description="Value(s) to filter. For continent: 'EU', 'NA', etc. For country: 'US', 'DE', etc. For airline: 'AA', 'LH', etc. For multiple values, use comma-separated (e.g., 'EU,AS')"
    )
    sql_query: Optional[str] = Field(
        default=None,
        description="Custom SQL query to filter routes_merged table (only used when filter_type='sql'). Must return columns compatible with routes_merged."
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional description of this filtered dataset"
    )


class ClearTempDatasetInput(BaseModel):
    """Input for clearing the temporary dataset."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    confirm: bool = Field(
        default=True,
        description="Confirmation to clear temp dataset"
    )


class GetTempDatasetStatusInput(BaseModel):
    """Input for getting temp dataset status."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    pass


class GetDatasetSchemaInput(BaseModel):
    """Input for retrieving dataset schema information."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    table_name: Optional[str] = Field(
        default=None,
        description="Specific table name to get schema for. If None, returns all table schemas. Options: airports, routes, routes_merged, navaids, airlines, countries"
    )


# =============================================================================
# Tools
# =============================================================================

@mcp.tool(
    name="aviation_create_temp_dataset",
    annotations={
        "title": "Create Filtered Temporary Dataset",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False
    }
)
async def create_temp_dataset(params: CreateTempDatasetInput) -> str:
    """Create a filtered temporary dataset for focused analysis.
    
    This tool allows you to filter the global aviation dataset to create a temporary
    working dataset. Once created, all analysis tools will use the filtered dataset
    instead of the global one until you clear it with aviation_clear_temp_dataset.
    
    Use cases:
    - Analyze only European routes
    - Focus on a specific airline's network
    - Study routes within certain countries
    - Create custom filtered datasets via SQL
    
    Args:
        params: CreateTempDatasetInput containing:
            - filter_type: Type of filter (continent/country/airline/sql)
            - filter_value: Value(s) to filter by
            - sql_query: Optional custom SQL query
            - description: Optional description
    
    Returns:
        Confirmation with statistics about the filtered dataset.
    """
    _ensure_data_loaded()
    
    global temp_dataset_active, temp_airports, temp_routes_dat, temp_routes_merged
    global temp_G, temp_filter_description
    
    # Build filter description
    if params.description:
        filter_desc = params.description
    else:
        filter_desc = f"{params.filter_type}: {params.filter_value}"
    
    # Apply filter based on type
    if params.filter_type == 'continent':
        continents = [c.strip().upper() for c in params.filter_value.split(',')]
        
        # Filter airports
        temp_airports = airports[airports['continent'].isin(continents)].copy()
        
        # Filter routes to only include airports in this continent
        temp_routes_merged = routes_merged[
            routes_merged['src_continent'].isin(continents) | 
            routes_merged['dst_continent'].isin(continents)
        ].copy()
        
        temp_routes_dat = routes_dat[
            routes_dat['Source'].isin(temp_airports.index) &
            routes_dat['Dest'].isin(temp_airports.index)
        ].copy()
        
    elif params.filter_type == 'country':
        countries_list = [c.strip().upper() for c in params.filter_value.split(',')]
        
        # Filter airports
        temp_airports = airports[airports['iso_country'].isin(countries_list)].copy()
        
        # Filter routes
        temp_routes_merged = routes_merged[
            routes_merged['src_country'].isin(countries_list) |
            routes_merged['dst_country'].isin(countries_list)
        ].copy()
        
        temp_routes_dat = routes_dat[
            routes_dat['Source'].isin(temp_airports.index) &
            routes_dat['Dest'].isin(temp_airports.index)
        ].copy()
        
    elif params.filter_type == 'airline':
        airlines_list = [a.strip().upper() for a in params.filter_value.split(',')]
        
        # Filter routes by airline
        temp_routes_dat = routes_dat[routes_dat['Airline'].isin(airlines_list)].copy()
        temp_routes_merged = routes_merged[routes_merged['Airline'].isin(airlines_list)].copy()
        
        # Filter airports to only those used by these routes
        airport_codes = set(temp_routes_dat['Source'].unique()) | set(temp_routes_dat['Dest'].unique())
        temp_airports = airports[airports.index.isin(airport_codes)].copy()
        
    elif params.filter_type == 'sql':
        if not params.sql_query:
            return "Error: sql_query parameter required when filter_type='sql'"
        
        try:
            # Execute custom SQL on routes_merged
            temp_routes_merged = pd.read_sql_query(params.sql_query, db_conn)
            
            # Extract routes_dat columns
            route_cols = ['Airline', 'AirlineID', 'Source', 'SourceID', 'Dest', 'DestID', 'Codeshare', 'Stops', 'Equipment']
            temp_routes_dat = temp_routes_merged[route_cols].copy() if all(col in temp_routes_merged.columns for col in route_cols) else routes_dat[routes_dat['Source'].isin(temp_routes_merged['Source']) & routes_dat['Dest'].isin(temp_routes_merged['Dest'])].copy()
            
            # Filter airports
            airport_codes = set(temp_routes_merged['Source'].unique()) | set(temp_routes_merged['Dest'].unique())
            temp_airports = airports[airports.index.isin(airport_codes)].copy()
            
        except Exception as e:
            return f"Error executing SQL query: {str(e)}"
    
    else:
        return f"Error: Unknown filter_type '{params.filter_type}'. Use: continent, country, airline, or sql"
    
    # Build graph from filtered data
    temp_G = _create_graph_from_routes(temp_airports, temp_routes_dat)
    
    # Activate temp dataset
    temp_dataset_active = True
    temp_filter_description = filter_desc
    
    # Generate statistics
    result = {
        "status": "success",
        "filter_description": filter_desc,
        "filter_type": params.filter_type,
        "filter_value": params.filter_value,
        "statistics": {
            "airports": len(temp_airports),
            "routes": len(temp_routes_dat),
            "graph_nodes": temp_G.number_of_nodes(),
            "graph_edges": temp_G.number_of_edges(),
            "continents": temp_airports['continent'].nunique(),
            "countries": temp_airports['iso_country'].nunique()
        }
    }
    
    md = "# Temporary Dataset Created ✓\n\n"
    md += f"**Filter:** {filter_desc}\n\n"
    md += "## Dataset Statistics\n\n"
    md += f"- **Airports:** {result['statistics']['airports']:,}\n"
    md += f"- **Routes:** {result['statistics']['routes']:,}\n"
    md += f"- **Graph Nodes:** {result['statistics']['graph_nodes']:,}\n"
    md += f"- **Graph Edges:** {result['statistics']['graph_edges']:,}\n"
    md += f"- **Continents:** {result['statistics']['continents']}\n"
    md += f"- **Countries:** {result['statistics']['countries']}\n\n"
    md += "## What This Means\n\n"
    md += "🔹 All analysis tools will now use this filtered dataset\n\n"
    md += "🔹 Tools affected:\n"
    md += "  - aviation_get_hubs\n"
    md += "  - aviation_analyze_countries\n"
    md += "  - aviation_find_shortest_path\n"
    md += "  - aviation_find_communities\n"
    md += "  - aviation_analyze_airline\n"
    md += "  - All other analysis tools\n\n"
    md += "🔹 To return to the full global dataset, use: `aviation_clear_temp_dataset`\n\n"
    md += "🔹 To check current status, use: `aviation_get_temp_dataset_status`\n"
    
    return md


@mcp.tool(
    name="aviation_clear_temp_dataset",
    annotations={
        "title": "Clear Temporary Dataset",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def clear_temp_dataset(params: ClearTempDatasetInput) -> str:
    """Clear the temporary dataset and return to using the global dataset.
    
    After calling this, all analysis tools will use the full global dataset again.
    
    Args:
        params: ClearTempDatasetInput with confirmation
    
    Returns:
        Confirmation message.
    """
    _ensure_data_loaded()
    
    global temp_dataset_active, temp_airports, temp_routes_dat, temp_routes_merged
    global temp_G, temp_filter_description
    
    if not temp_dataset_active:
        return "ℹ️ No temporary dataset is currently active. Already using global dataset."
    
    # Clear temp dataset
    old_description = temp_filter_description
    temp_dataset_active = False
    temp_airports = None
    temp_routes_dat = None
    temp_routes_merged = None
    temp_G = None
    temp_filter_description = None
    
    md = "# Temporary Dataset Cleared ✓\n\n"
    md += f"**Previous Filter:** {old_description}\n\n"
    md += "## Status\n\n"
    md += "✅ Now using the full global dataset\n\n"
    md += "All analysis tools will now operate on:\n"
    md += f"- **Airports:** {len(airports):,}\n"
    md += f"- **Routes:** {len(routes_dat):,}\n"
    md += f"- **Countries:** {airports['iso_country'].nunique()}\n"
    md += f"- **Continents:** {airports['continent'].nunique()}\n"
    
    return md


@mcp.tool(
    name="aviation_get_temp_dataset_status",
    annotations={
        "title": "Get Temporary Dataset Status",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def get_temp_dataset_status(params: GetTempDatasetStatusInput) -> str:
    """Get information about the current dataset being used.
    
    Shows whether a temporary filtered dataset is active or if using the global dataset.
    
    Returns:
        Status information about the active dataset.
    """
    _ensure_data_loaded()
    
    datasets = _get_active_datasets()
    
    md = "# Dataset Status\n\n"
    
    if datasets['is_temp']:
        md += "## 🔹 Temporary Filtered Dataset Active\n\n"
        md += f"**Filter:** {datasets['description']}\n\n"
        md += "### Current Dataset Statistics\n\n"
        md += f"- **Airports:** {len(datasets['airports']):,}\n"
        md += f"- **Routes:** {len(datasets['routes_dat']):,}\n"
        md += f"- **Graph Nodes:** {datasets['graph'].number_of_nodes():,}\n"
        md += f"- **Graph Edges:** {datasets['graph'].number_of_edges():,}\n"
        md += f"- **Continents:** {datasets['airports']['continent'].nunique()}\n"
        md += f"- **Countries:** {datasets['airports']['iso_country'].nunique()}\n\n"
        md += "### Comparison with Global Dataset\n\n"
        md += f"- **Global Airports:** {len(airports):,}\n"
        md += f"- **Global Routes:** {len(routes_dat):,}\n"
        md += f"- **Filtered to:** {len(datasets['airports'])/len(airports)*100:.1f}% of airports, {len(datasets['routes_dat'])/len(routes_dat)*100:.1f}% of routes\n\n"
        md += "To clear this filter and return to global dataset, use: `aviation_clear_temp_dataset`\n"
    else:
        md += "## 🌍 Using Global Dataset\n\n"
        md += "No temporary filter is active.\n\n"
        md += "### Global Dataset Statistics\n\n"
        md += f"- **Airports:** {len(datasets['airports']):,}\n"
        md += f"- **Routes:** {len(datasets['routes_dat']):,}\n"
        md += f"- **Graph Nodes:** {datasets['graph'].number_of_nodes():,}\n"
        md += f"- **Graph Edges:** {datasets['graph'].number_of_edges():,}\n"
        md += f"- **Continents:** {datasets['airports']['continent'].nunique()}\n"
        md += f"- **Countries:** {datasets['airports']['iso_country'].nunique()}\n\n"
        md += "To create a filtered dataset, use: `aviation_create_temp_dataset`\n"
    
    return md


@mcp.tool(
    name="aviation_get_hubs",
    annotations={
        "title": "Get Top Hub Airports",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def get_hubs(params: GetHubsInput) -> str:
    """Find the top hub airports by number of flight connections.
    
    Analyzes the flight network to identify airports with the most
    connections. Can filter by continent and uses temp dataset if active.
    
    Args:
        params: GetHubsInput containing:
            - top_n: Number of hubs to return (default 10)
            - continent_filter: Optional continent code filter
            - response_format: Output format (markdown/json/csv)
    
    Returns:
        Hub rankings with connection counts and locations.
    """
    _ensure_data_loaded()
    
    # Get active dataset
    datasets = _get_active_datasets()
    airports_df = datasets['airports']
    G = datasets['graph']
    
    # Filter by continent if specified
    if params.continent_filter:
        filtered_airports = airports_df[airports_df['continent'] == params.continent_filter]
        G = G.subgraph(filtered_airports.index).copy()
        airports_df = filtered_airports
    
    # Calculate hub scores
    scores = dict(G.degree())
    df = pd.DataFrame.from_dict(scores, orient='index', columns=['Connections'])
    df = df.join(airports_df[['name', 'iso_country', 'continent', 'latitude_deg', 'longitude_deg']], how='left')
    df = df.sort_values('Connections', ascending=False)
    top_hubs = df.head(params.top_n)
    
    result = {
        "hubs": [],
        "total_airports": len(airports_df),
        "total_routes": G.number_of_edges(),
        "filter": params.continent_filter,
        "using_temp_dataset": datasets['is_temp'],
        "dataset_description": datasets['description']
    }
    
    for code, row in top_hubs.iterrows():
        result["hubs"].append({
            "code": code,
            "name": row.get('name', 'Unknown'),
            "country": row.get('iso_country', 'Unknown'),
            "continent": row.get('continent', 'Unknown'),
            "connections": int(row['Connections']),
            "latitude": row.get('latitude_deg'),
            "longitude": row.get('longitude_deg')
        })
    
    if params.response_format == ResponseFormat.CSV:
        csv_df = pd.DataFrame(result["hubs"])
        return csv_df.to_csv(index=False)
    
    if params.response_format == ResponseFormat.JSON:
        return json.dumps(result, indent=2)
    
    # Markdown format
    md = f"# Top {params.top_n} Hub Airports\n\n"
    if datasets['is_temp']:
        md += f"🔹 **Using Filtered Dataset:** {datasets['description']}\n\n"
    if params.continent_filter:
        md += f"**Additional Filter:** {params.continent_filter}\n\n"
    md += f"**Total Airports:** {result['total_airports']:,} | **Total Routes:** {result['total_routes']:,}\n\n"
    md += "| Rank | Code | Airport Name | Country | Connections |\n"
    md += "|------|------|--------------|---------|-------------|\n"
    for i, hub in enumerate(result["hubs"], 1):
        md += f"| {i} | {hub['code']} | {hub['name'][:40]} | {hub['country']} | {hub['connections']:,} |\n"
    
    return md


@mcp.tool(
    name="aviation_get_transcontinental_hubs",
    annotations={
        "title": "Get Transcontinental Hub Airports",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def get_transcontinental_hubs(params: GetTranscontinentalHubsInput) -> str:
    """Find hub airports connecting two continents.
    
    Identifies airports with the most flight connections between two specific
    continents, useful for understanding intercontinental travel patterns.
    
    Args:
        params: GetTranscontinentalHubsInput containing:
            - continent_a: First continent code (e.g., 'NA')
            - continent_b: Second continent code (e.g., 'EU')
            - top_n: Number of hubs to return
    
    Returns:
        Rankings of airports connecting the two continents with connection counts.
    """
    _ensure_data_loaded()
    
    # Count connections between continents
    traffic = {}
    for u, v in G_global.edges():
        try:
            c_u = airports.loc[u, 'continent']
            c_v = airports.loc[v, 'continent']
            if {c_u, c_v} == {params.continent_a, params.continent_b}:
                traffic[u] = traffic.get(u, 0) + 1
                traffic[v] = traffic.get(v, 0) + 1
        except KeyError:
            continue
    
    df = pd.DataFrame.from_dict(traffic, orient='index', columns=['Connections'])
    df = df.join(airports[['name', 'iso_country', 'continent', 'latitude_deg', 'longitude_deg']], how='left')
    df = df.sort_values('Connections', ascending=False)
    top_hubs = df.head(params.top_n)
    
    result = {
        "continent_a": params.continent_a,
        "continent_b": params.continent_b,
        "hubs": []
    }
    
    for code, row in top_hubs.iterrows():
        result["hubs"].append({
            "code": code,
            "name": row.get('name', 'Unknown'),
            "country": row.get('iso_country', 'Unknown'),
            "continent": row.get('continent', 'Unknown'),
            "connections": int(row['Connections'])
        })
    
    if params.response_format == ResponseFormat.JSON:
        return json.dumps(result, indent=2)
    
    md = f"# Top {params.top_n} {params.continent_a}-{params.continent_b} Hub Airports\n\n"
    md += "| Rank | Code | Airport Name | Country | Continent | Connections |\n"
    md += "|------|------|--------------|---------|-----------|-------------|\n"
    for i, hub in enumerate(result["hubs"], 1):
        md += f"| {i} | {hub['code']} | {hub['name'][:35]} | {hub['country']} | {hub['continent']} | {hub['connections']} |\n"
    
    return md


@mcp.tool(
    name="aviation_analyze_countries",
    annotations={
        "title": "Analyze Country Flight Connectivity",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def analyze_countries(params: CountryAnalysisInput) -> str:
    """Analyze flight connectivity metrics by country.
    
    Calculates domestic, international, and intercontinental flight statistics
    for each country, with options to sort by different metrics. Uses temp dataset if active.
    
    Args:
        params: CountryAnalysisInput containing:
            - top_n: Number of countries to show
            - metric: Sorting metric (total, domestic, international, etc.)
    
    Returns:
        Country rankings with flight volume and connectivity rates.
    """
    _ensure_data_loaded()
    
    # Get active dataset
    datasets = _get_active_datasets()
    df = datasets['routes_merged'].copy()
    
    df['is_domestic'] = df['src_country'] == df['dst_country']
    df['is_intercontinental'] = df['src_continent'] != df['dst_continent']
    df['is_intl'] = ~df['is_domestic']
    
    country_stats = df.groupby('src_country').agg(
        Total_Flights=('Source', 'count'),
        Domestic_Flights=('is_domestic', 'sum'),
        International_Flights=('is_intl', 'sum'),
        Intercontinental_Flights=('is_intercontinental', 'sum')
    ).astype(int)
    
    country_stats['Domestic_Rate'] = country_stats['Domestic_Flights'] / country_stats['Total_Flights']
    country_stats['Intl_Rate'] = country_stats['International_Flights'] / country_stats['Total_Flights']
    country_stats['Intercontinental_Rate'] = country_stats['Intercontinental_Flights'] / country_stats['Total_Flights']
    
    # Add country names
    if not countries.empty:
        country_stats = country_stats.join(countries.rename(columns={'name': 'Country_Name'}), how='left')
    else:
        country_stats['Country_Name'] = country_stats.index
    
    # Sort by requested metric
    metric_map = {
        'total': 'Total_Flights',
        'domestic': 'Domestic_Flights',
        'international': 'International_Flights',
        'intercontinental': 'Intercontinental_Flights',
        'domestic_rate': 'Domestic_Rate',
        'international_rate': 'Intl_Rate',
        'intercontinental_rate': 'Intercontinental_Rate'
    }
    sort_col = metric_map.get(params.metric, 'Total_Flights')
    top_countries = country_stats.sort_values(sort_col, ascending=False).head(params.top_n)
    
    result = {
        "metric": params.metric,
        "countries": [],
        "using_temp_dataset": datasets['is_temp'],
        "dataset_description": datasets['description']
    }
    
    for code, row in top_countries.iterrows():
        result["countries"].append({
            "code": code,
            "name": str(row.get('Country_Name', code)),
            "total_flights": int(row['Total_Flights']),
            "domestic_flights": int(row['Domestic_Flights']),
            "international_flights": int(row['International_Flights']),
            "intercontinental_flights": int(row['Intercontinental_Flights']),
            "domestic_rate": round(float(row['Domestic_Rate']), 3),
            "international_rate": round(float(row['Intl_Rate']), 3),
            "intercontinental_rate": round(float(row['Intercontinental_Rate']), 3)
        })
    
    if params.response_format == ResponseFormat.JSON:
        return json.dumps(result, indent=2)
    
    md = f"# Top {params.top_n} Countries by {params.metric.replace('_', ' ').title()}\n\n"
    if datasets['is_temp']:
        md += f"🔹 **Using Filtered Dataset:** {datasets['description']}\n\n"
    md += "| Rank | Country | Total | Domestic | International | Intercontinental | Dom% | Intl% | Inter% |\n"
    md += "|------|---------|-------|----------|---------------|------------------|------|-------|--------|\n"
    for i, c in enumerate(result["countries"], 1):
        md += f"| {i} | {c['name'][:20]} ({c['code']}) | {c['total_flights']:,} | {c['domestic_flights']:,} | {c['international_flights']:,} | {c['intercontinental_flights']:,} | {c['domestic_rate']:.1%} | {c['international_rate']:.1%} | {c['intercontinental_rate']:.1%} |\n"
    
    return md


@mcp.tool(
    name="aviation_find_communities",
    annotations={
        "title": "Find Flight Communities",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def find_communities(params: FlightCommunitiesInput) -> str:
    """Identify world flight communities using Louvain algorithm.
    
    Detects clusters of airports that are densely interconnected by flights,
    revealing regional flight patterns and travel hubs.
    
    Args:
        params: FlightCommunitiesInput containing:
            - top_n: Number of top communities to detail
    
    Returns:
        Community analysis with member airports, geographic composition, and hub info.
    """
    _ensure_data_loaded()
    
    G_undirected = G_global.to_undirected()
    largest_cc = max(nx.connected_components(G_undirected), key=len)
    G_cc = G_undirected.subgraph(largest_cc).copy()
    
    communities = nx_comm.louvain_communities(G_cc, seed=42, resolution=1.0)
    
    community_data = []
    for idx, comm in enumerate(sorted(communities, key=len, reverse=True)):
        comm_airports = airports[airports.index.isin(comm)].copy()
        if len(comm_airports) == 0:
            continue
        
        country_counts = comm_airports['iso_country'].value_counts()
        continent_counts = comm_airports['continent'].value_counts()
        
        subgraph = G_cc.subgraph(comm)
        degrees = dict(subgraph.degree())
        
        if degrees:
            hub = max(degrees, key=degrees.get)
            hub_name = airports.loc[hub, 'name'] if hub in airports.index else hub
        else:
            hub = list(comm)[0]
            hub_name = hub
        
        community_data.append({
            'id': idx,
            'size': len(comm),
            'airports': list(comm),
            'top_countries': country_counts.head(5).to_dict(),
            'continents': continent_counts.to_dict(),
            'hub': hub,
            'hub_name': hub_name,
            'avg_lat': float(comm_airports['latitude_deg'].mean()),
            'avg_lon': float(comm_airports['longitude_deg'].mean())
        })
    
    result = {
        "total_communities": len(community_data),
        "communities": community_data[:params.top_n]
    }
    
    if params.response_format == ResponseFormat.JSON:
        return json.dumps(result, indent=2, default=str)
    
    md = f"# World Flight Communities Analysis\n\n"
    md += f"**Total Communities Found:** {result['total_communities']}\n\n"
    
    for i, comm in enumerate(result["communities"], 1):
        md += f"## Community {i}\n"
        md += f"- **Size:** {comm['size']} airports\n"
        md += f"- **Hub:** {comm['hub']} ({comm['hub_name'][:40]})\n"
        md += f"- **Top Countries:** {', '.join([f'{k} ({v})' for k, v in list(comm['top_countries'].items())[:3]])}\n"
        md += f"- **Continents:** {', '.join([f'{k} ({v})' for k, v in comm['continents'].items()])}\n\n"
    
    return md


@mcp.tool(
    name="aviation_find_outliers",
    annotations={
        "title": "Find Geographic Outliers",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def find_outliers(params: OutliersInput) -> str:
    """Find geographic outlier airports in a region.
    
    Identifies the northernmost, southernmost, easternmost, westernmost, 
    and highest elevation airports in a specified continent or country.
    
    Args:
        params: OutliersInput containing:
            - region_filter: Region code (e.g., 'EU', 'DE')
            - region_name: Human-readable name
            - filter_type: 'continent' or 'country'
    
    Returns:
        Geographic outlier airports with their locations and details.
    """
    _ensure_data_loaded()
    
    if params.filter_type == 'continent':
        region_airports = airports[airports['continent'] == params.region_filter].copy()
    else:
        region_airports = airports[airports['iso_country'] == params.region_filter].copy()
    
    if len(region_airports) == 0:
        return f"No airports found for {params.region_name} ({params.region_filter})"
    
    outliers = {}
    
    # Northernmost
    northernmost = region_airports.loc[region_airports['latitude_deg'].idxmax()]
    outliers['Northernmost'] = {
        'code': northernmost.name,
        'name': northernmost['name'],
        'country': northernmost['iso_country'],
        'latitude': northernmost['latitude_deg'],
        'longitude': northernmost['longitude_deg'],
        'elevation': northernmost.get('elevation_ft', 'N/A')
    }
    
    # Southernmost
    southernmost = region_airports.loc[region_airports['latitude_deg'].idxmin()]
    outliers['Southernmost'] = {
        'code': southernmost.name,
        'name': southernmost['name'],
        'country': southernmost['iso_country'],
        'latitude': southernmost['latitude_deg'],
        'longitude': southernmost['longitude_deg'],
        'elevation': southernmost.get('elevation_ft', 'N/A')
    }
    
    # Easternmost
    easternmost = region_airports.loc[region_airports['longitude_deg'].idxmax()]
    outliers['Easternmost'] = {
        'code': easternmost.name,
        'name': easternmost['name'],
        'country': easternmost['iso_country'],
        'latitude': easternmost['latitude_deg'],
        'longitude': easternmost['longitude_deg'],
        'elevation': easternmost.get('elevation_ft', 'N/A')
    }
    
    # Westernmost
    westernmost = region_airports.loc[region_airports['longitude_deg'].idxmin()]
    outliers['Westernmost'] = {
        'code': westernmost.name,
        'name': westernmost['name'],
        'country': westernmost['iso_country'],
        'latitude': westernmost['latitude_deg'],
        'longitude': westernmost['longitude_deg'],
        'elevation': westernmost.get('elevation_ft', 'N/A')
    }
    
    # Highest elevation
    if 'elevation_ft' in region_airports.columns:
        region_airports_elev = region_airports.dropna(subset=['elevation_ft'])
        if len(region_airports_elev) > 0:
            highest = region_airports_elev.loc[region_airports_elev['elevation_ft'].idxmax()]
            outliers['Highest'] = {
                'code': highest.name,
                'name': highest['name'],
                'country': highest['iso_country'],
                'latitude': highest['latitude_deg'],
                'longitude': highest['longitude_deg'],
                'elevation': highest['elevation_ft']
            }
    
    result = {
        "region": params.region_name,
        "filter_code": params.region_filter,
        "filter_type": params.filter_type,
        "total_airports": len(region_airports),
        "outliers": outliers
    }
    
    if params.response_format == ResponseFormat.JSON:
        return json.dumps(result, indent=2)
    
    md = f"# Geographic Outliers: {params.region_name}\n\n"
    md += f"**Region Code:** {params.region_filter} | **Total Airports:** {result['total_airports']}\n\n"
    
    for direction, data in outliers.items():
        md += f"## {direction}\n"
        md += f"- **Airport:** {data['name']} ({data['code']})\n"
        md += f"- **Country:** {data['country']}\n"
        md += f"- **Coordinates:** {data['latitude']:.4f}°, {data['longitude']:.4f}°\n"
        md += f"- **Elevation:** {data['elevation']} ft\n\n"
    
    return md


@mcp.tool(
    name="aviation_nearest_neighbors",
    annotations={
        "title": "Find Nearest Airports/Navaids",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def nearest_neighbors(params: NearestNeighborsInput) -> str:
    """Find nearest airports and navigation aids to a location.
    
    Given an airport code or coordinates, finds the k nearest airports
    and navaids using great circle distance.
    
    Args:
        params: NearestNeighborsInput containing:
            - query: Airport IATA code or 'lat,lon' coordinates
            - k: Number of nearest items to find
    
    Returns:
        Lists of nearest airports and navaids with distances.
    """
    _ensure_data_loaded()
    
    # Parse query
    if ',' in params.query:
        # Coordinates
        try:
            lat, lon = map(float, params.query.split(','))
            query_point = np.array([[np.radians(lat), np.radians(lon)]])
            query_name = f"Coordinates {lat:.4f}, {lon:.4f}"
        except ValueError:
            return "Error: Invalid coordinate format. Use 'lat,lon' (e.g., '51.47,-0.45')"
    else:
        # Airport code
        code = params.query.upper()
        if code not in airports.index:
            return f"Error: Airport code '{code}' not found"
        airport_data = airports.loc[code]
        lat = airport_data['latitude_deg']
        lon = airport_data['longitude_deg']
        query_point = np.array([[np.radians(lat), np.radians(lon)]])
        query_name = f"{airport_data['name']} ({code})"
    
    # Find nearest airports
    airport_coords = np.radians(airports[['latitude_deg', 'longitude_deg']].values)
    tree_airports = BallTree(airport_coords, metric='haversine')
    distances, indices = tree_airports.query(query_point, k=params.k + 1)
    distances = distances[0] * EARTH_RADIUS_KM
    indices = indices[0]
    
    nearest_airports = []
    for i, (dist, idx) in enumerate(zip(distances[1:], indices[1:]), 1):
        airport = airports.iloc[idx]
        nearest_airports.append({
            'rank': i,
            'code': airport.name,
            'name': airport['name'],
            'country': airport['iso_country'],
            'distance_km': round(float(dist), 2)
        })
    
    # Find nearest navaids
    navaid_coords = np.radians(navaids[['latitude_deg', 'longitude_deg']].values)
    tree_navaids = BallTree(navaid_coords, metric='haversine')
    distances_nav, indices_nav = tree_navaids.query(query_point, k=params.k)
    distances_nav = distances_nav[0] * EARTH_RADIUS_KM
    indices_nav = indices_nav[0]
    
    nearest_navaids = []
    for i, (dist, idx) in enumerate(zip(distances_nav, indices_nav), 1):
        navaid = navaids.iloc[idx]
        nearest_navaids.append({
            'rank': i,
            'name': navaid.get('name', 'Unknown'),
            'type': navaid.get('type', 'Unknown'),
            'distance_km': round(float(dist), 2)
        })
    
    result = {
        "query": params.query,
        "query_name": query_name,
        "nearest_airports": nearest_airports,
        "nearest_navaids": nearest_navaids
    }
    
    if params.response_format == ResponseFormat.JSON:
        return json.dumps(result, indent=2)
    
    md = f"# Nearest Neighbors Analysis\n\n"
    md += f"**Query Point:** {query_name}\n\n"
    
    md += f"## Nearest {params.k} Airports\n\n"
    md += "| Rank | Code | Airport Name | Country | Distance (km) |\n"
    md += "|------|------|--------------|---------|---------------|\n"
    for airport in nearest_airports:
        md += f"| {airport['rank']} | {airport['code']} | {airport['name'][:35]} | {airport['country']} | {airport['distance_km']:,.2f} |\n"
    
    md += f"\n## Nearest {params.k} Navigation Aids\n\n"
    md += "| Rank | Name | Type | Distance (km) |\n"
    md += "|------|------|------|---------------|\n"
    for navaid in nearest_navaids:
        md += f"| {navaid['rank']} | {navaid['name'][:30]} | {navaid['type']} | {navaid['distance_km']:,.2f} |\n"
    
    return md


@mcp.tool(
    name="aviation_analyze_airline",
    annotations={
        "title": "Analyze Airline Operations",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def analyze_airline(params: AirlineAnalysisInput) -> str:
    """Analyze airline operations and route network.
    
    Provides detailed statistics about an airline's routes, destinations,
    and aircraft equipment usage.
    
    Args:
        params: AirlineAnalysisInput containing:
            - airline_code: Airline IATA code
            - top_n: Number of top items to show
    
    Returns:
        Airline operational statistics and top destinations/equipment.
    """
    _ensure_data_loaded()
    
    airline_routes = routes_dat[routes_dat['Airline'] == params.airline_code]
    
    if len(airline_routes) == 0:
        return f"No routes found for airline '{params.airline_code}'"
    
    # Get airline info
    airline_name = "Unknown"
    if params.airline_code in airlines.index:
        airline_info = airlines.loc[params.airline_code]
        airline_name = airline_info.get('Name', 'Unknown')
    
    # Top destinations
    dest_counts = airline_routes['Dest'].value_counts().head(params.top_n)
    top_destinations = []
    for dest, count in dest_counts.items():
        if dest in airports.index:
            airport = airports.loc[dest]
            top_destinations.append({
                'code': dest,
                'name': airport['name'],
                'country': airport['iso_country'],
                'routes': int(count)
            })
    
    # Top origins
    origin_counts = airline_routes['Source'].value_counts().head(params.top_n)
    top_origins = []
    for origin, count in origin_counts.items():
        if origin in airports.index:
            airport = airports.loc[origin]
            top_origins.append({
                'code': origin,
                'name': airport['name'],
                'country': airport['iso_country'],
                'routes': int(count)
            })
    
    # Equipment analysis
    equipment_list = []
    for equip_str in airline_routes['Equipment'].dropna():
        equipment_list.extend(str(equip_str).split())
    equipment_counts = pd.Series(equipment_list).value_counts().head(params.top_n)
    
    top_equipment = [{'type': equip, 'count': int(count)} 
                     for equip, count in equipment_counts.items()]
    
    result = {
        "airline_code": params.airline_code,
        "airline_name": airline_name,
        "total_routes": len(airline_routes),
        "unique_origins": airline_routes['Source'].nunique(),
        "unique_destinations": airline_routes['Dest'].nunique(),
        "top_destinations": top_destinations,
        "top_origins": top_origins,
        "top_equipment": top_equipment
    }
    
    if params.response_format == ResponseFormat.JSON:
        return json.dumps(result, indent=2)
    
    md = f"# Airline Analysis: {airline_name} ({params.airline_code})\n\n"
    md += f"**Total Routes:** {result['total_routes']:,} | "
    md += f"**Unique Origins:** {result['unique_origins']:,} | "
    md += f"**Unique Destinations:** {result['unique_destinations']:,}\n\n"
    
    md += f"## Top {params.top_n} Destinations\n\n"
    md += "| Rank | Code | Airport Name | Country | Routes |\n"
    md += "|------|------|--------------|---------|--------|\n"
    for i, dest in enumerate(top_destinations, 1):
        md += f"| {i} | {dest['code']} | {dest['name'][:35]} | {dest['country']} | {dest['routes']} |\n"
    
    md += f"\n## Top {params.top_n} Origins\n\n"
    md += "| Rank | Code | Airport Name | Country | Routes |\n"
    md += "|------|------|--------------|---------|--------|\n"
    for i, origin in enumerate(top_origins, 1):
        md += f"| {i} | {origin['code']} | {origin['name'][:35]} | {origin['country']} | {origin['routes']} |\n"
    
    md += f"\n## Top {params.top_n} Aircraft Types\n\n"
    md += "| Rank | Equipment Type | Count |\n"
    md += "|------|----------------|-------|\n"
    for i, equip in enumerate(top_equipment, 1):
        md += f"| {i} | {equip['type']} | {equip['count']} |\n"
    
    return md


@mcp.tool(
    name="aviation_analyze_equipment",
    annotations={
        "title": "Analyze Aircraft Equipment",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def analyze_equipment(params: EquipmentAnalysisInput) -> str:
    """Analyze aircraft equipment usage across all routes.
    
    Provides statistics on the most commonly used aircraft types
    in the global route network.
    
    Args:
        params: EquipmentAnalysisInput containing:
            - top_n: Number of top aircraft types to show
    
    Returns:
        Rankings of aircraft equipment with usage counts.
    """
    _ensure_data_loaded()
    
    # Parse all equipment
    equipment_list = []
    for equip_str in routes_dat['Equipment'].dropna():
        equipment_list.extend(str(equip_str).split())
    
    equipment_counts = pd.Series(equipment_list).value_counts().head(params.top_n)
    
    result = {
        "total_equipment_entries": len(equipment_list),
        "unique_types": len(set(equipment_list)),
        "top_equipment": []
    }
    
    for equip_type, count in equipment_counts.items():
        result["top_equipment"].append({
            "type": equip_type,
            "count": int(count),
            "percentage": round(100 * count / len(equipment_list), 2)
        })
    
    if params.response_format == ResponseFormat.JSON:
        return json.dumps(result, indent=2)
    
    md = f"# Aircraft Equipment Analysis\n\n"
    md += f"**Total Equipment Entries:** {result['total_equipment_entries']:,} | "
    md += f"**Unique Types:** {result['unique_types']:,}\n\n"
    md += f"## Top {params.top_n} Aircraft Types\n\n"
    md += "| Rank | Equipment Type | Count | Percentage |\n"
    md += "|------|----------------|-------|------------|\n"
    for i, equip in enumerate(result["top_equipment"], 1):
        md += f"| {i} | {equip['type']} | {equip['count']:,} | {equip['percentage']}% |\n"
    
    return md


@mcp.tool(
    name="aviation_get_airport_info",
    annotations={
        "title": "Get Airport Information",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def get_airport_info(params: AirportInfoInput) -> str:
    """Get detailed information about a specific airport.
    
    Provides airport details including location, size, and optionally
    its route network statistics.
    
    Args:
        params: AirportInfoInput containing:
            - airport_code: Airport IATA code
            - include_routes: Whether to include route info
    
    Returns:
        Comprehensive airport information.
    """
    _ensure_data_loaded()
    
    code = params.airport_code.upper()
    if code not in airports.index:
        return f"Error: Airport code '{code}' not found"
    
    airport = airports.loc[code]
    
    result = {
        "code": code,
        "name": airport['name'],
        "city": airport.get('municipality', 'Unknown'),
        "country": airport['iso_country'],
        "continent": airport['continent'],
        "latitude": float(airport['latitude_deg']),
        "longitude": float(airport['longitude_deg']),
        "elevation_ft": airport.get('elevation_ft', 'N/A'),
        "iata_code": airport.get('iata_code', code),
        "icao_code": airport.get('ident', 'N/A')
    }
    
    if params.include_routes and code in G_global:
        out_degree = G_global.out_degree(code)
        in_degree = G_global.in_degree(code)
        
        result["route_stats"] = {
            "outbound_routes": out_degree,
            "inbound_routes": in_degree,
            "total_connections": out_degree + in_degree
        }
    
    if params.response_format == ResponseFormat.JSON:
        return json.dumps(result, indent=2)
    
    md = f"# Airport Information: {result['name']}\n\n"
    md += f"**IATA Code:** {result['code']} | **ICAO Code:** {result['icao_code']}\n\n"
    md += f"## Location\n"
    md += f"- **City:** {result['city']}\n"
    md += f"- **Country:** {result['country']}\n"
    md += f"- **Continent:** {result['continent']}\n"
    md += f"- **Coordinates:** {result['latitude']:.4f}°, {result['longitude']:.4f}°\n"
    md += f"- **Elevation:** {result['elevation_ft']} ft\n\n"
    
    if "route_stats" in result:
        md += f"## Route Network\n"
        md += f"- **Outbound Routes:** {result['route_stats']['outbound_routes']}\n"
        md += f"- **Inbound Routes:** {result['route_stats']['inbound_routes']}\n"
        md += f"- **Total Connections:** {result['route_stats']['total_connections']}\n"
    
    return md



@mcp.tool(
    name="aviation_find_shortest_path",
    annotations={
        "title": "Find Shortest Flight Path",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def find_shortest_path(params: ShortestPathInput) -> str:
    """Find the shortest flight path between two airports.
    
    Uses NetworkX shortest path algorithm to find the route with the minimum
    number of connections (stops) between any two airports. Uses temp dataset if active.
    
    Args:
        params: ShortestPathInput containing:
            - start_airport: Origin airport IATA code
            - end_airport: Destination airport IATA code
            - response_format: Output format (markdown/json/csv)
    
    Returns:
        The shortest path with stops, airport details, and distance information.
    """
    _ensure_data_loaded()
    
    # Get active dataset
    datasets = _get_active_datasets()
    airports_df = datasets['airports']
    G = datasets['graph']
    
    start = params.start_airport.upper()
    end = params.end_airport.upper()
    
    # Validate airports exist
    if start not in airports.index:
        return f"Error: Starting airport '{start}' not found in database"
    if end not in airports.index:
        return f"Error: Destination airport '{end}' not found in database"
    
    # Check if airports are in the active dataset
    if start not in airports_df.index:
        return f"Error: Airport '{start}' not in current dataset (Filter: {datasets['description']})"
    if end not in airports_df.index:
        return f"Error: Airport '{end}' not in current dataset (Filter: {datasets['description']})"
    
    # Check if airports are in the graph
    if start not in G:
        return f"Error: Airport '{start}' has no connections in the route network"
    if end not in G:
        return f"Error: Airport '{end}' has no connections in the route network"
    
    # Find shortest path
    try:
        path = nx.shortest_path(G, start, end)
        stops = len(path) - 1
        
        # Calculate total distance
        total_distance = 0
        path_details = []
        
        for i in range(len(path)):
            airport_code = path[i]
            airport_data = airports.loc[airport_code]
            
            leg_distance = None
            if i < len(path) - 1:
                next_code = path[i + 1]
                if G.has_edge(airport_code, next_code):
                    leg_distance = G[airport_code][next_code].get('weight', 0)
                    total_distance += leg_distance
            
            path_details.append({
                'step': i + 1,
                'airport_code': airport_code,
                'airport_name': airport_data['name'],
                'city': airport_data.get('municipality', 'Unknown'),
                'country': airport_data['iso_country'],
                'continent': airport_data['continent'],
                'latitude': float(airport_data['latitude_deg']),
                'longitude': float(airport_data['longitude_deg']),
                'leg_distance_km': round(float(leg_distance), 2) if leg_distance else None
            })
        
        result = {
            "start_airport": start,
            "start_name": airports.loc[start]['name'],
            "end_airport": end,
            "end_name": airports.loc[end]['name'],
            "path_found": True,
            "total_stops": stops,
            "total_legs": len(path) - 1,
            "total_distance_km": round(float(total_distance), 2) if total_distance > 0 else None,
            "path": path,
            "path_details": path_details,
            "using_temp_dataset": datasets['is_temp'],
            "dataset_description": datasets['description']
        }
        
    except nx.NetworkXNoPath:
        result = {
            "start_airport": start,
            "start_name": airports.loc[start]['name'],
            "end_airport": end,
            "end_name": airports.loc[end]['name'],
            "path_found": False,
            "message": f"No route exists between {start} and {end} in the current network",
            "using_temp_dataset": datasets['is_temp'],
            "dataset_description": datasets['description']
        }
    
    # Format output
    if params.response_format == ResponseFormat.CSV:
        if result["path_found"]:
            df = pd.DataFrame(result["path_details"])
            return df.to_csv(index=False)
        else:
            return "No path found"
    
    if params.response_format == ResponseFormat.JSON:
        return json.dumps(result, indent=2)
    
    # Markdown format
    if not result["path_found"]:
        md = f"# No Route Found\n\n"
        if result.get('using_temp_dataset'):
            md += f"🔹 **Using Filtered Dataset:** {result['dataset_description']}\n\n"
        md += f"**From:** {result['start_name']} ({result['start_airport']})\n\n"
        md += f"**To:** {result['end_name']} ({result['end_airport']})\n\n"
        md += f"⚠️ {result['message']}\n\n"
        md += "This could mean:\n"
        md += "- The airports are in disconnected components of the network\n"
        md += "- No airline operates routes connecting these regions\n"
        md += "- The route requires multiple booking systems (not in dataset)\n"
        if result.get('using_temp_dataset'):
            md += "- The filtered dataset may not include all necessary connections\n"
        return md
    
    md = f"# Shortest Flight Path\n\n"
    if result.get('using_temp_dataset'):
        md += f"🔹 **Using Filtered Dataset:** {result['dataset_description']}\n\n"
    md += f"**From:** {result['start_name']} ({result['start_airport']})\n\n"
    md += f"**To:** {result['end_name']} ({result['end_airport']})\n\n"
    md += f"**Total Stops:** {result['total_stops']}\n\n"
    md += f"**Total Legs:** {result['total_legs']}\n\n"
    
    if result['total_distance_km']:
        md += f"**Total Distance:** {result['total_distance_km']:,.2f} km ({result['total_distance_km'] * 0.621371:,.2f} miles)\n\n"
    
    md += "## Route Path\n\n"
    md += "```\n"
    md += " → ".join(result['path'])
    md += "\n```\n\n"
    
    md += "## Detailed Itinerary\n\n"
    md += "| Step | Airport | City | Country | Continent | Leg Distance (km) |\n"
    md += "|------|---------|------|---------|-----------|-------------------|\n"
    
    for detail in result['path_details']:
        leg_dist = f"{detail['leg_distance_km']:,.2f}" if detail['leg_distance_km'] else "-"
        md += f"| {detail['step']} | **{detail['airport_code']}** {detail['airport_name'][:25]} | {detail['city'][:20]} | {detail['country']} | {detail['continent']} | {leg_dist} |\n"
    
    md += "\n## Summary Statistics\n\n"
    
    # Calculate some stats
    continents_visited = set(d['continent'] for d in result['path_details'])
    countries_visited = set(d['country'] for d in result['path_details'])
    
    md += f"- **Continents Visited:** {len(continents_visited)} ({', '.join(sorted(continents_visited))})\n"
    md += f"- **Countries Visited:** {len(countries_visited)} ({', '.join(sorted(countries_visited)[:5])}"
    if len(countries_visited) > 5:
        md += f" + {len(countries_visited) - 5} more"
    md += ")\n"
    md += f"- **Average Leg Distance:** {result['total_distance_km'] / result['total_legs']:,.2f} km\n" if result['total_distance_km'] else ""
    
    return md


@mcp.tool(
    name="aviation_sql_query",
    annotations={
        "title": "Execute SQL Query on Aviation Data",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def sql_query(params: SQLQueryInput) -> str:
    """Execute custom SQL queries on aviation datasets.
    
    Allows flexible data analysis using SQL on the following tables:
    - airports: Airport information (iata_code, name, iso_country, continent, latitude_deg, longitude_deg, elevation_ft, etc.)
    - routes: Flight routes (Airline, Source, Dest, Equipment, Stops, Codeshare)
    - routes_merged: Routes with geographic info (includes src_country, dst_country, src_continent, dst_continent)
    - navaids: Navigation aids (name, type, latitude_deg, longitude_deg)
    - airlines: Airline information (IATA, Name, Country, Active)
    - countries: Country information (code, name)
    
    Examples:
    - Filter EU routes: "SELECT * FROM routes_merged WHERE src_continent = 'EU' AND dst_continent = 'EU'"
    - Count by airline: "SELECT Airline, COUNT(*) as routes FROM routes GROUP BY Airline ORDER BY routes DESC"
    - Find specific equipment: "SELECT * FROM routes WHERE Equipment LIKE '%737%'"
    
    Args:
        params: SQLQueryInput containing:
            - query: SQL query to execute
            - limit: Maximum rows to return (default 100)
            - response_format: Output format (markdown/json/csv)
    
    Returns:
        Query results in the requested format.
    """
    _ensure_data_loaded()
    
    try:
        # Add LIMIT if not already present (case-insensitive check)
        query_upper = params.query.upper()
        if 'LIMIT' not in query_upper:
            query_with_limit = f"{params.query} LIMIT {params.limit}"
        else:
            query_with_limit = params.query
        
        # Execute query
        df = pd.read_sql_query(query_with_limit, db_conn)
        
        if len(df) == 0:
            return "Query returned no results."
        
        result = {
            "query": params.query,
            "rows_returned": len(df),
            "columns": list(df.columns),
            "data": df.to_dict('records')
        }
        
        if params.response_format == ResponseFormat.CSV:
            return df.to_csv(index=False)
        
        if params.response_format == ResponseFormat.JSON:
            return json.dumps(result, indent=2, default=str)
        
        # Markdown format
        md = f"# SQL Query Results\n\n"
        md += f"**Query:** `{params.query}`\n\n"
        md += f"**Rows Returned:** {len(df)}\n\n"
        
        # Create markdown table
        md += "| " + " | ".join(df.columns) + " |\n"
        md += "|" + "|".join(["---"] * len(df.columns)) + "|\n"
        
        for _, row in df.iterrows():
            md += "| " + " | ".join([str(val)[:50] for val in row.values]) + " |\n"
        
        if len(df) == params.limit:
            md += f"\n*Results limited to {params.limit} rows. Use the 'limit' parameter to see more.*\n"
        
        return md
        
    except Exception as e:
        return f"Error executing query: {str(e)}\n\nAvailable tables: airports, routes, routes_merged, navaids, airlines, countries"


@mcp.tool(
    name="aviation_get_dataset_schema",
    annotations={
        "title": "Get Dataset Schema Information",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def get_dataset_schema(params: GetDatasetSchemaInput) -> str:
    """Get schema information for aviation datasets.
    
    Returns column names and types for all or a specific table,
    helping with SQL query construction.
    
    Args:
        params: GetDatasetSchemaInput containing:
            - table_name: Optional specific table name
    
    Returns:
        Schema information for the requested table(s).
    """
    _ensure_data_loaded()
    
    tables = {
        'airports': airports.reset_index(),
        'routes': routes_dat,
        'routes_merged': routes_merged,
        'navaids': navaids,
        'airlines': airlines.reset_index(),
        'countries': countries.reset_index() if not countries.empty else pd.DataFrame()
    }
    
    if params.table_name:
        if params.table_name not in tables:
            return f"Error: Table '{params.table_name}' not found. Available tables: {', '.join(tables.keys())}"
        tables = {params.table_name: tables[params.table_name]}
    
    md = "# Aviation Dataset Schema\n\n"
    
    for table_name, df in tables.items():
        if len(df) == 0:
            continue
            
        md += f"## Table: {table_name}\n\n"
        md += f"**Rows:** {len(df):,} | **Columns:** {len(df.columns)}\n\n"
        md += "| Column Name | Data Type | Sample Values |\n"
        md += "|-------------|-----------|---------------|\n"
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            # Get sample non-null values
            sample_vals = df[col].dropna().head(3).tolist()
            sample_str = ', '.join([str(v)[:30] for v in sample_vals])
            md += f"| {col} | {dtype} | {sample_str} |\n"
        
        md += "\n"
    
    md += "## Usage Examples\n\n"
    md += "```sql\n"
    md += "-- Find all routes in Europe\n"
    md += "SELECT * FROM routes_merged WHERE src_continent = 'EU'\n\n"
    md += "-- Count routes by airline\n"
    md += "SELECT Airline, COUNT(*) as total FROM routes GROUP BY Airline ORDER BY total DESC\n\n"
    md += "-- Find airports in a specific country\n"
    md += "SELECT iata_code, name, elevation_ft FROM airports WHERE iso_country = 'US'\n\n"
    md += "-- Analyze equipment usage\n"
    md += "SELECT Equipment, COUNT(*) as count FROM routes WHERE Equipment != '' GROUP BY Equipment\n"
    md += "```\n"
    
    return md


if __name__ == "__main__":
    mcp.run()