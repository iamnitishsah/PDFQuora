import os
import sys
from dotenv import load_dotenv

load_dotenv()


def check_environment():
    required_vars = ['PINECONE_API_KEY', 'GOOGLE_API_KEY']
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        print(f"Error: Missing environment variables: {', '.join(missing)}")
        print("Please set them in your .env file")
        return False
    return True


def show_menu():
    print("\n" + "=" * 50)
    print("           PDF QUORA")
    print("    Ask Questions About Your PDF")
    print("=" * 50)
    print("\n1. Upload and Process PDF")
    print("2. Ask Questions")
    print("3. Exit")
    print("-" * 30)


def main():
    if not check_environment():
        return

    while True:
        show_menu()
        choice = input("Select an option (1-3): ").strip()

        if choice == '1':
            print("\n📄 Processing PDF Document...")
            try:
                from ingest_document import main as ingest_main
                ingest_main()
            except ImportError:
                print("Error: ingest_document.py not found")
            except Exception as e:
                print(f"Error during PDF processing: {str(e)}")

        elif choice == '2':
            print("\n🤖 Starting Q&A Session...")
            try:
                from answer_generator import main as answer_main
                answer_main()
            except ImportError:
                print("Error: answer_generator.py not found")
            except Exception as e:
                print(f"Error during Q&A: {str(e)}")

        elif choice == '3':
            print("\nThank you for using PDF Quora! 👋")
            break

        else:
            print("Invalid choice. Please select 1, 2, or 3.")


if __name__ == "__main__":
    main()