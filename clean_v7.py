def clean_v7_file():
    """Remove empty lines and ensure only unique lines remain in v7.txt"""
    try:
        # Read the file and collect unique non-empty lines
        unique_lines = set()
        
        with open('data/v7.txt', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Only keep non-empty lines
                    unique_lines.add(line)
        
        # Write back the cleaned content
        with open('data/v7.txt', 'w', encoding='utf-8') as f:
            for line in sorted(unique_lines):
                f.write(f"{line}\n")
        
        print(f"Cleaned v7.txt: {len(unique_lines)} unique non-empty lines")
        
    except Exception as e:
        print(f"Error cleaning file: {e}")

if __name__ == "__main__":
    clean_v7_file()


