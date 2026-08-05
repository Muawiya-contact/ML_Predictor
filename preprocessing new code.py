# New from  wasiq.
import re
from typing import Dict, List, Tuple, Optional, Set
from difflib import get_close_matches
import string

class MedicalComplaintPreprocessor:
    def __init__(self):
        # Initialize all dictionaries
        self.init_dictionaries()
        
    def init_dictionaries(self):
        """Initialize all required dictionaries for the pipeline"""
        
        # Standard English medical terms dictionary
        self.english_medical_dict = {
            # Symptoms
            'headache': 'headache',
            'migraine': 'migraine',
            'fever': 'fever',
            'cough': 'cough',
            'cold': 'cold',
            'vomiting': 'vomiting',
            'diarrhea': 'diarrhea',
            'nausea': 'nausea',
            'weakness': 'weakness',
            'dizziness': 'dizziness',
            'constipation': 'constipation',
            'pain': 'pain',
            'ache': 'ache',
            'swelling': 'swelling',
            'infection': 'infection',
            'bleeding': 'bleeding',
            'fracture': 'fracture',
            
            # Body parts
            'head': 'head',
            'stomach': 'stomach',
            'chest': 'chest',
            'back': 'back',
            'neck': 'neck',
            'shoulder': 'shoulder',
            'arm': 'arm',
            'leg': 'leg',
            'knee': 'knee',
            'hand': 'hand',
            'foot': 'foot',
            
            # Intensities
            'severe': 'severe',
            'mild': 'mild',
            'moderate': 'moderate',
            'acute': 'acute',
            'chronic': 'chronic'
        }
        
        # Standard Roman Urdu dictionary (correct spellings)
        self.roman_urdu_dict = {
            # Symptoms
            'dard': 'pain',
            'bukhar': 'fever',
            'khansi': 'cough',
            'zukaam': 'cold',
            'nazla': 'cold',
            'qay': 'vomiting',
            'ulti': 'vomiting',
            'dast': 'diarrhea',
            'kamzori': 'weakness',
            'chakkar': 'dizziness',
            'qabz': 'constipation',
            'soojan': 'swelling',
            'matli': 'nausea',
            'behoshi': 'unconsciousness',
            'khoon': 'bleeding',
            
            # Body parts
            'sir': 'head',
            'sar': 'head',
            'pet': 'stomach',
            'peet': 'stomach',
            'seenay': 'chest',
            'kamar': 'back',
            'gardan': 'neck',
            'kandha': 'shoulder',
            'bazoo': 'arm',
            'baazu': 'arm',
            'pair': 'leg',
            'taang': 'leg',
            'ghutna': 'knee',
            'haath': 'hand',
            'paon': 'foot',
            
            # Intensities
            'bahut': 'severe',
            'bohat': 'severe',
            'bohot': 'severe',
            'tez': 'severe',
            'shadeed': 'severe',
            'thoda': 'mild',
            'halka': 'mild'
        }
        
        # Stop words for removal
        self.stop_words = {
            'hai', 'hain', 'tha', 'thi', 'the', 'ho', 'hai', 'hein',
            'mera', 'meri', 'mere', 'tera', 'teri', 'apna', 'apni',
            'ko', 'se', 'ka', 'ki', 'ke', 'ne', 'mein', 'par', 'tak',
            'aur', 'ya', 'lekin', 'magar', 'to', 'agar', 'raha', 'rahi',
            'kar', 'karta', 'karti', 'ye', 'wo', 'vo', 'main', 'hum',
            'aap', 'tum', 'mujhe', 'tumhe', 'is', 'the', 'a', 'an'
        }
        
        # Common misspellings mapping (for fuzzy correction)
        self.common_misspellings = {
            # English misspellings
            'hedache': 'headache',
            'headake': 'headache',
            'headche': 'headache',
            'stomack': 'stomach',
            'stomic': 'stomach',
            'chestpain': 'chest pain',
            'backpain': 'back pain',
            'fevar': 'fever',
            'feaver': 'fever',
            'caugh': 'cough',
            'coughing': 'cough',
            'vomitting': 'vomiting',
            'diharrea': 'diarrhea',
            'diareha': 'diarrhea',
            'nausia': 'nausea',
            'weaknesss': 'weakness',
            'diziness': 'dizziness',
            
            # Roman Urdu misspellings
            'dardd': 'dard',
            'dardh': 'dard',
            'bukhaar': 'bukhar',
            'bukharh': 'bukhar',
            'khanshi': 'khansi',
            'khansii': 'khansi',
            'zukaamh': 'zukaam',
            'nazlaa': 'nazla',
            'qayy': 'qay',
            'ultii': 'ulti',
            'dastt': 'dast',
            'kamzoori': 'kamzori',
            'kamzori': 'kamzori',
            'chakker': 'chakkar',
            'chakkar': 'chakkar',
            'qabzz': 'qabz',
            'soojan': 'soojan',
            'matlii': 'matli'
        }
        
        # Multi-word mappings
        self.multi_word_mappings = {
            'sir dard': 'headache',
            'sar dard': 'headache',
            'pet dard': 'stomach pain',
            'peet dard': 'stomach pain',
            'seenay ka dard': 'chest pain',
            'kamar dard': 'back pain',
            'gardan dard': 'neck pain',
            'sans ki takleef': 'breathing difficulty',
            'haddi tut gayi': 'fracture',
            'khoon ana': 'bleeding',
            'behoshi ana': 'unconsciousness'
        }
        
        # For fuzzy matching
        self.english_words_list = list(self.english_medical_dict.keys())
        self.roman_urdu_words_list = list(self.roman_urdu_dict.keys())
    
    def preprocess_chief_complaint(self, chief_complaint: str) -> Dict:
        """
        Main preprocessing pipeline as per professor's flowchart
        """
        print("\n" + "="*80)
        print("📋 CHIEF COMPLAINT PREPROCESSING PIPELINE")
        print("="*80)
        print(f"INPUT: {chief_complaint}")
        
        result = {
            'original_text': chief_complaint,
            'english_portion': None,
            'roman_urdu_portion': None,
            'stop_words_removed': None,
            'corrected_english': None,
            'corrected_roman_urdu': None,
            'fuzzy_mapped_english': None,
            'mapped_english_representation': None,
            'truncated_chief_complaint': None,
            'processing_steps': {}
        }
        
        # Step 1: Split English and Roman Urdu portions
        english_part, roman_part = self.split_english_urdu(chief_complaint)
        result['english_portion'] = english_part
        result['roman_urdu_portion'] = roman_part
        
        result['processing_steps']['Step1_Split'] = {
            'description': 'Split English and Roman Urdu portions',
            'english': english_part,
            'roman_urdu': roman_part
        }
        
        # Step 2: Remove stop words from both portions
        english_no_stop = self.remove_stop_words(english_part)
        roman_no_stop = self.remove_stop_words(roman_part)
        result['stop_words_removed'] = {
            'english': english_no_stop,
            'roman_urdu': roman_no_stop
        }
        
        result['processing_steps']['Step2_StopWords'] = {
            'description': 'Remove stop words',
            'english_after': english_no_stop,
            'roman_after': roman_no_stop
        }
        
        # Step 3: Fuzzy mapping for English portion (correct spellings)
        corrected_english = self.fuzzy_map_english(english_no_stop)
        result['corrected_english'] = corrected_english
        result['fuzzy_mapped_english'] = corrected_english
        
        result['processing_steps']['Step3_FuzzyEnglish'] = {
            'description': 'Fuzzy mapping to correct English spellings',
            'original': english_no_stop,
            'corrected': corrected_english
        }
        
        # Step 4: Correct Roman Urdu spellings
        corrected_roman = self.correct_roman_urdu_spelling(roman_no_stop)
        result['corrected_roman_urdu'] = corrected_roman
        
        result['processing_steps']['Step4_CorrectRoman'] = {
            'description': 'Correct Roman Urdu spellings',
            'original': roman_no_stop,
            'corrected': corrected_roman
        }
        
        # Step 5: Fuzzy mapping Roman Urdu to English
        mapped_from_roman = self.fuzzy_map_roman_to_english(corrected_roman)
        
        # Step 6: Combine both portions
        combined_mapping = self.combine_mappings(corrected_english, mapped_from_roman)
        result['mapped_english_representation'] = combined_mapping
        
        result['processing_steps']['Step5_MapRomanToEnglish'] = {
            'description': 'Fuzzy mapping Roman Urdu to English',
            'roman_text': corrected_roman,
            'mapped': mapped_from_roman
        }
        
        # Step 7: Create truncated chief complaint (simplified version)
        result['truncated_chief_complaint'] = self.create_truncated_complaint(combined_mapping)
        
        result['processing_steps']['Step6_Combine'] = {
            'description': 'Combine English and mapped Roman portions',
            'combined': combined_mapping,
            'truncated': result['truncated_chief_complaint']
        }
        
        # Print detailed processing
        self.print_processing_steps(result)
        
        return result
    
    def split_english_urdu(self, text: str) -> Tuple[str, str]:
        """
        Split text into English and Roman Urdu portions
        Based on known English words vs Roman Urdu words
        """
        words = text.lower().split()
        english_words = []
        roman_words = []
        
        for word in words:
            # Check if word is likely English
            if word in self.english_medical_dict or word in self.english_words_list:
                english_words.append(word)
            # Check if it's common English words like 'left', 'right'
            elif word in ['left', 'right', 'both', 'upper', 'lower', 'my', 'your', 'the']:
                english_words.append(word)
            else:
                roman_words.append(word)
        
        english_part = ' '.join(english_words)
        roman_part = ' '.join(roman_words)
        
        return english_part, roman_part
    
    def remove_stop_words(self, text: str) -> str:
        """Remove stop words from text"""
        if not text:
            return ""
        
        words = text.lower().split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)
    
    def fuzzy_map_english(self, text: str, cutoff: float = 0.8) -> str:
        """
        Fuzzy mapping to correct English spellings
        Uses difflib to find closest matches
        """
        if not text:
            return ""
        
        words = text.split()
        corrected_words = []
        
        for word in words:
            # Check for common misspellings first
            if word in self.common_misspellings:
                corrected_words.append(self.common_misspellings[word])
                continue
            
            # If word is already correct, keep it
            if word in self.english_medical_dict or word in self.english_words_list:
                corrected_words.append(word)
                continue
            
            # Try fuzzy matching
            matches = get_close_matches(word, self.english_words_list, n=1, cutoff=cutoff)
            if matches:
                corrected_words.append(matches[0])
            else:
                # Keep original if no match found
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def correct_roman_urdu_spelling(self, text: str, cutoff: float = 0.8) -> str:
        """
        Step 4: Correct Roman Urdu spellings using dictionary and fuzzy matching
        """
        if not text:
            return ""
        
        words = text.split()
        corrected_words = []
        
        for word in words:
            # Check for common misspellings first
            if word in self.common_misspellings:
                corrected_words.append(self.common_misspellings[word])
                continue
            
            # If word is already correct, keep it
            if word in self.roman_urdu_dict:
                corrected_words.append(word)
                continue
            
            # Try fuzzy matching
            matches = get_close_matches(word, self.roman_urdu_words_list, n=1, cutoff=cutoff)
            if matches:
                corrected_words.append(matches[0])
            else:
                # Keep original if no match found
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def fuzzy_map_roman_to_english(self, text: str, cutoff: float = 0.8) -> str:
        """
        Step 5: Fuzzy mapping Roman Urdu words to English
        """
        if not text:
            return ""
        
        words = text.split()
        english_mappings = []
        
        # First, check for multi-word mappings
        remaining_text = text
        for phrase, mapping in self.multi_word_mappings.items():
            if phrase in remaining_text:
                english_mappings.append(mapping)
                remaining_text = remaining_text.replace(phrase, '')
        
        # Process remaining words
        remaining_words = remaining_text.split()
        for word in remaining_words:
            # Direct mapping
            if word in self.roman_urdu_dict:
                english_mappings.append(self.roman_urdu_dict[word])
            else:
                # Try fuzzy matching against Roman Urdu dictionary
                matches = get_close_matches(word, self.roman_urdu_words_list, n=1, cutoff=cutoff)
                if matches:
                    english_mappings.append(self.roman_urdu_dict[matches[0]])
                else:
                    # If still no match, keep original as is
                    english_mappings.append(word)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_mappings = []
        for word in english_mappings:
            if word not in seen:
                seen.add(word)
                unique_mappings.append(word)
        
        return ' '.join(unique_mappings)
    
    def combine_mappings(self, english_text: str, roman_mapped_text: str) -> str:
        """
        Combine English portion and mapped Roman Urdu portion
        Remove duplicates
        """
        english_words = english_text.split() if english_text else []
        roman_words = roman_mapped_text.split() if roman_mapped_text else []
        
        # Combine words
        all_words = english_words + roman_words
        
        # Remove duplicates while preserving order
        seen = set()
        unique_words = []
        for word in all_words:
            if word.lower() not in seen:
                seen.add(word.lower())
                unique_words.append(word)
        
        return ' '.join(unique_words)
    
    def create_truncated_complaint(self, combined_text: str, max_words: int = 7) -> str:
        """
        Create truncated version of chief complaint (key symptoms only)
        """
        if not combined_text:
            return ""
        
        # Priority keywords to keep
        priority_keywords = {
            'pain', 'severe', 'fever', 'vomiting', 'diarrhea', 'bleeding',
            'fracture', 'unconscious', 'headache', 'chest', 'breathing'
        }
        
        words = combined_text.split()
        
        # Filter to keep only priority keywords and important words
        important_words = []
        for word in words:
            if word.lower() in priority_keywords or word in self.english_medical_dict:
                important_words.append(word)
        
        # If we have too many words, truncate
        if len(important_words) > max_words:
            important_words = important_words[:max_words]
        
        return ' '.join(important_words) if important_words else combined_text
    
    def print_processing_steps(self, result: Dict):
        """
        Print detailed processing steps matching professor's flowchart
        """
        print("\n" + "="*80)
        print("📊 PROCESSING STEPS (As per Flowchart)")
        print("="*80)
        
        steps = result['processing_steps']
        
        for step_name, step_info in steps.items():
            print(f"\n🔹 {step_name}: {step_info['description']}")
            print(f"   → {step_info}")
        
        print("\n" + "="*80)
        print("✅ FINAL RESULTS")
        print("="*80)
        print(f"📝 Original: {result['original_text']}")
        print(f"🔧 English Portion: {result['english_portion']}")
        print(f"🔧 Roman Urdu Portion: {result['roman_urdu_portion']}")
        print(f"🎯 Final Mapping: {result['mapped_english_representation']}")
        print(f"📌 Truncated Complaint: {result['truncated_chief_complaint']}")
        print("="*80)
    
    def batch_process(self, complaints: List[str]) -> List[Dict]:
        """Process multiple chief complaints"""
        results = []
        for complaint in complaints:
            result = self.preprocess_chief_complaint(complaint)
            results.append(result)
        return results


def visualize_flowchart():
    """Visualize the preprocessing pipeline"""
    print("\n" + "="*80)
    print("🔄 PREPROCESSING PIPELINE FLOWCHART")
    print("="*80)
    print("""
    INPUT: Chief Complaint in Roman Urdu/English
                    │
                    ▼
    ┌──────────────────────────────────┐
    │  Step 1: Split English & Roman   │
    │         Urdu Portions             │
    └──────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
    English Portion         Roman Urdu Portion
        │                       │
        ▼                       ▼
    Remove Stop Words       Remove Stop Words
        │                       │
        ▼                       ▼
    Fuzzy Mapping to        Correct Roman
    Correct English         Urdu Spellings
    Spellings                   │
        │                       ▼
        │                   Fuzzy Mapping to
        │                   English Dictionary
        │                       │
        └───────────┬───────────┘
                    ▼
        ┌──────────────────────────────────┐
        │  Step 6: Combine Mappings        │
        │  - Remove duplicates              │
        │  - Create final representation   │
        └──────────────────────────────────┘
                    │
                    ▼
        ┌──────────────────────────────────┐
        │  Step 7: Create Truncated        │
        │  Chief Complaint (Simplified)    │
        └──────────────────────────────────┘
                    │
                    ▼
           FINAL ENGLISH REPRESENTATION
    """)


# Test the implementation
def test_preprocessor():
    """Test the preprocessing pipeline with sample inputs"""
    preprocessor = MedicalComplaintPreprocessor()
    
    test_cases = [
        "Mera sir bahut dard kar raha hai",
        "Left bazoo mein severe pain hai",
        "Mujhe bukhar aur khansi hai",
        "Right pair mein dard",
        "Chest pain and seenay mein dard",
        "My head is paining since yesterday",
        "Haddi tut gayi right hand mein",
        "Meri kamar mein tez dard aur kamzori hai"
    ]
    
    print("\n" + "="*80)
    print("🧪 TESTING PREPROCESSING PIPELINE")
    print("="*80)
    
    results = preprocessor.batch_process(test_cases)
    
    # Print summary
    print("\n\n" + "="*80)
    print("📊 SUMMARY OF RESULTS")
    print("="*80)
    print(f"{'Input':<30} → {'Output':<30}")
    print("-"*80)
    
    for result in results:
        input_text = result['original_text'][:27] + "..." if len(result['original_text']) > 30 else result['original_text']
        output_text = result['truncated_chief_complaint'][:27] + "..." if len(result['truncated_chief_complaint']) > 30 else result['truncated_chief_complaint']
        print(f"{input_text:<30} → {output_text:<30}")
    
    return results


if __name__ == "__main__":
    # Visualize the flowchart
    visualize_flowchart()
    
    # Run tests
    results = test_preprocessor()
    
    # Interactive mode
    print("\n" + "="*80)
    print("💬 INTERACTIVE MODE")
    print("="*80)
    print("Enter 'exit' to quit\n")
    
    preprocessor = MedicalComplaintPreprocessor()
    
    while True:
        text = input("\nEnter Chief Complaint: ").strip()
        if text.lower() in ['exit', 'quit', 'q']:
            print("\n👋 Exiting...")
            break
        
        if text:
            result = preprocessor.preprocess_chief_complaint(text)
        else:
            print("⚠️ Please enter some text!")