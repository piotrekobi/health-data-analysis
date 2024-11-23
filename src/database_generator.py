import sqlite3
import pandas as pd
from pathlib import Path
import logging
from typing import Optional, Union, List
import contextlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HealthDataDB:
    def __init__(self):
        self.conn = None
        self.cursor = None
        
    def initialize_db(self):
        """Create an in-memory SQLite database and load the CSV data"""
        try:
            # Create in-memory database
            self.conn = sqlite3.connect(':memory:')
            self.cursor = self.conn.cursor()
            
            # Load CSVs into pandas first to handle data types properly
            df1 = pd.read_csv('data/health_dataset1.csv')
            df2 = pd.read_csv('data/health_dataset2.csv')
            
            # Write DataFrames to SQLite
            df1.to_sql('health', self.conn, index=False)
            df2.to_sql('physical_activity', self.conn, index=False)
            
            logger.info("Database initialized successfully")
            
            # Create indices for better query performance
            self.cursor.execute('CREATE INDEX idx_patient_number1 ON health(Patient_Number)')
            self.cursor.execute('CREATE INDEX idx_patient_number2 ON physical_activity(Patient_Number)')
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute a SQL query and return results as a DataFrame"""
        try:
            if not self.conn:
                raise RuntimeError("Database not initialized. Call initialize_db() first.")
            
            result = pd.read_sql_query(query, self.conn)
            logger.info(f"Query executed successfully. Returned {len(result)} rows.")
            return result
            
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise
    
    def get_table_info(self, table_name: str) -> List[tuple]:
        """Get column information for a table"""
        try:
            if not self.conn:
                raise RuntimeError("Database not initialized. Call initialize_db() first.")
            
            self.cursor.execute(f"PRAGMA table_info({table_name})")
            return self.cursor.fetchall()
            
        except Exception as e:
            logger.error(f"Error getting table info: {str(e)}")
            raise
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
