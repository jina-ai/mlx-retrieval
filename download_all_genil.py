import requests
import csv
from io import StringIO

def download_and_process_genil_language(lang_code):
    """Download and process a specific language file from GeniL dataset"""
    url = f"https://raw.githubusercontent.com/google-research-datasets/GeniL/refs/heads/main/dataset/{lang_code}.csv"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse CSV content
        csv_content = StringIO(response.text)
        reader = csv.DictReader(csv_content)
        
        # Get column names
        columns = reader.fieldnames
        
        # Collect unique values for each column
        unique_values = {}
        for col in columns:
            unique_values[col] = set()
        
        # Read all rows and collect unique values
        for row in reader:
            for col in columns:
                if row[col] and row[col].strip():  # Skip empty values
                    unique_values[col].add(row[col].strip())
        
        # Write to language-specific file
        output_file = f'data/v7_{lang_code}.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"=== GeniL Dataset - {lang_code.upper()} ===\n\n")
            for col in columns:
                f.write(f"=== {col} ===\n")
                for value in sorted(unique_values[col]):
                    f.write(f"{value}\n")
                f.write("\n")
        
        print(f"✅ {lang_code.upper()}: {len(columns)} columns, {sum(len(vals) for vals in unique_values.values())} total unique values")
        return True
        
    except requests.RequestException as e:
        print(f"❌ Error downloading {lang_code}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error processing {lang_code}: {e}")
        return False

def download_all_genil():
    """Download and process all available language files"""
    languages = ['ar', 'bn', 'en', 'es', 'fr', 'hi', 'id', 'ms', 'pt']
    
    print("Downloading and processing all GeniL language datasets...")
    print("=" * 60)
    
    success_count = 0
    for lang in languages:
        if download_and_process_genil_language(lang):
            success_count += 1
    
    print("=" * 60)
    print(f"Completed: {success_count}/{len(languages)} languages processed successfully")
    
    if success_count == len(languages):
        print("All language files have been downloaded and processed!")
        print("Check the 'data/' directory for v7_[lang].txt files")

if __name__ == "__main__":
    download_all_genil()


