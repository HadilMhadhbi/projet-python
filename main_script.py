"""
Main Script
Drug Consumption Prediction Project
"""

import os

def main():
    print("=" * 60)
    print("DRUG CONSUMPTION PREDICTION PROJECT")
    print("=" * 60)
    
    print("\n1. Scraping (Week 1)")
    print("2. EDA (Week 1)")
    print("3. Preprocessing (Week 2)")
    print("4. Exit")
    
    choice = input("\nSelect option: ").strip()
    
    if choice == '1':
        import scraping_
    elif choice == '2':
        import EDA
    elif choice == '3':
        import preprocessing
    elif choice == '4':
        print("Goodbye!")
    else:
        print("Invalid option")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    main()