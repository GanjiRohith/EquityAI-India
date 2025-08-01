#!/usr/bin/env python3
"""
Demo script for the Indian Stock Market Research Assistant
This script demonstrates the capabilities by running a sample analysis
"""

from stock_research_assistant import IndianStockResearchAssistant
import sys

def run_demo():
    """Run a demo analysis for a popular Indian stock"""
    
    print("🚀 Indian Stock Market Research Assistant - Demo")
    print("=" * 60)
    print()
    
    # Initialize the assistant
    assistant = IndianStockResearchAssistant()
    
    # Demo stock - RELIANCE (Reliance Industries)
    demo_stock = "RELIANCE"
    
    print(f"📊 Running demo analysis for {demo_stock}...")
    print("This will show you the comprehensive analysis capabilities.")
    print()
    
    try:
        # Generate the report
        report = assistant.generate_report(demo_stock)
        
        # Display the report
        print(report)
        
        # Save the report
        from datetime import datetime
        filename = f"demo_{demo_stock}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n💾 Demo report saved to: {filename}")
        print("\n✅ Demo completed successfully!")
        print("\nTo run your own analysis:")
        print("1. Command Line: python stock_research_assistant.py")
        print("2. Web Interface: streamlit run streamlit_app.py")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("Please check your internet connection and try again.")
        return False
    
    return True

def show_usage():
    """Show usage instructions"""
    print("\n📖 Usage Instructions:")
    print("=" * 40)
    print()
    print("1. **Install Dependencies:**")
    print("   pip install -r requirements.txt")
    print()
    print("2. **Run Web Interface (Recommended):**")
    print("   streamlit run streamlit_app.py")
    print()
    print("3. **Run Command Line Interface:**")
    print("   python stock_research_assistant.py")
    print()
    print("4. **Popular Indian Stocks to Try:**")
    print("   - RELIANCE (Reliance Industries)")
    print("   - TCS (Tata Consultancy Services)")
    print("   - INFY (Infosys)")
    print("   - HDFCBANK (HDFC Bank)")
    print("   - ICICIBANK (ICICI Bank)")
    print("   - SBIN (State Bank of India)")
    print()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        show_usage()
    else:
        success = run_demo()
        if success:
            print("\n🎉 Ready to analyze your favorite stocks!")
        else:
            print("\n❌ Demo failed. Please check the setup instructions.") 