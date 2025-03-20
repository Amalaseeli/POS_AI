import yaml
import pyodbc

class DatabaseConnector:
    def read_yaml(self):
        try:
            # Attempt to read database credentials from the YAML file
            with open('db_cred.yaml', 'r') as f:
                credentials = yaml.safe_load(f)
            return credentials
        except FileNotFoundError:
            print("Error: The 'db_cred.yaml' file is missing.")
            return None
        except yaml.YAMLError as e:
            print(f"Error: YAML parsing error - {e}")
            return None
    
    def ini_db_engine(self):
        credentials = self.read_yaml()
        
        # If the credentials are None (due to an error in reading YAML), return None
        if credentials is None:
            return None
        
        server = credentials.get("server")
        database = credentials.get("database")
        username = credentials.get("username")
        password = credentials.get("password")
        
        if not all([server, database, username, password]):
            print("Error: Missing database credentials.")
            return None
        
        try:
            # Establish the database connection using pyodbc
            conn = pyodbc.connect(f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}')
            print("Database connection successful.")
            return conn
        except pyodbc.Error as e:
            print(f"Error: Failed to connect to the database. {e}")
            return None

if __name__ == "__main__":
    db_connector = DatabaseConnector()
    connection = db_connector.ini_db_engine()
    
    if connection:
        cursor = connection.cursor()
        connection.close()
