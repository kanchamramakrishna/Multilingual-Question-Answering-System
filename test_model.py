"""
Model Testing Script
Quick tests to verify model is working correctly
"""

from app.model_loader import ModelLoader
from app.inference import QAInference


def test_english():
    """Test English question answering"""
    print("\n" + "=" * 80)
    print("🇬🇧 TESTING ENGLISH")
    print("=" * 80)
    
    test_cases = [
        {
            "question": "What is the capital of France?",
            "context": "Paris is the capital and most populous city of France.",
            "expected": "Paris"
        },
        {
            "question": "When was the Eiffel Tower built?",
            "context": "The Eiffel Tower was constructed from 1887 to 1889.",
            "expected": "1887 to 1889"
        }
    ]
    
    return test_cases


def test_german():
    """Test German question answering"""
    print("\n" + "=" * 80)
    print("🇩🇪 TESTING GERMAN")
    print("=" * 80)
    
    test_cases = [
        {
            "question": "Was ist die Hauptstadt von Deutschland?",
            "context": "Berlin ist die Hauptstadt von Deutschland.",
            "expected": "Berlin"
        },
        {
            "question": "Wann wurde der Berliner Fernsehturm gebaut?",
            "context": "Der Berliner Fernsehturm wurde zwischen 1965 und 1969 erbaut.",
            "expected": "1965 bis 1969"
        }
    ]
    
    return test_cases


def run_tests():
    """Run all tests"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        🧪 MODEL TESTING SUITE 🧪                             ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Load model
    print("\n📂 Loading model...")
    try:
        loader = ModelLoader(model_path="models/multilingual_model")
        model, tokenizer = loader.load()
        
        inference = QAInference(model, tokenizer, loader.device)
        print("✅ Model loaded successfully!\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("\n💡 Make sure model files exist in models/multilingual_model/")
        return
    
    # Test English
    english_tests = test_english()
    passed = 0
    total = len(english_tests)
    
    for i, test in enumerate(english_tests, 1):
        answer, _ = inference.answer_question(
            test["question"], 
            test["context"], 
            "English"
        )
        
        print(f"\nTest {i}/{total}")
        print(f"Q: {test['question']}")
        print(f"Expected: {test['expected']}")
        print(f"Got: {answer}")
        
        if test["expected"].lower() in answer.lower():
            print("✅ PASSED")
            passed += 1
        else:
            print("❌ FAILED")
    
    print(f"\n📊 English Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
    
    # Test German
    german_tests = test_german()
    passed = 0
    total = len(german_tests)
    
    for i, test in enumerate(german_tests, 1):
        answer, _ = inference.answer_question(
            test["question"], 
            test["context"], 
            "German"
        )
        
        print(f"\nTest {i}/{total}")
        print(f"Q: {test['question']}")
        print(f"Expected: {test['expected']}")
        print(f"Got: {answer}")
        
        if test["expected"].lower() in answer.lower():
            print("✅ PASSED")
            passed += 1
        else:
            print("❌ FAILED")
    
    print(f"\n📊 German Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("✅ TESTING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()