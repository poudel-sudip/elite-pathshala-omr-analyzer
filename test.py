import requests
import json
import sys

def test_health():
    """Test API health endpoint"""
    print("[*] Testing API health endpoint...\n")

    url = 'http://localhost:8000/health'

    try:
        response = requests.get(url)
        result = response.json()

        print(f"Status Code: {response.status_code}")
        print(json.dumps(result, indent=2))

        if result.get('status') == 'healthy':
            print("✓ API is healthy!\n")
            return True
        else:
            print("✗ API health check failed\n")
            return False

    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to server on port 5000")
        print("Make sure Flask is running: python app.py\n")
        return False

def test_local_omr():
    """Test OMR analysis with OMR sheet URL"""
    print("[*] Testing OMR analysis with sample image...\n")

    url = 'http://localhost:8000/api/analyze-omr'
    data = {
        'omr_sheet': 'https://dev-storage.elitepathshala.com/uploads/omr-exam/7/6jiMcv4LYG4muS2aReJAjkX4AQepYsQaae5x5xZc.jpg',
        # 'correct_answers' : {'1':'A'}
    }

    try:
        response = requests.post(url, json=data)
        result = response.json()

        print(f"Status Code: {response.status_code}\n")
        print("Response:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        if result.get('success'):
            print("\n✓ OMR Analysis Successful!")
            print(f"  Student ID: {result.get('data')['student_id']}")
            # print(f"  Answers Detected: {len(result.get('answers', {}))}")
            # if result.get('answers'):
                # print(f"  Sample Answers: Q1={result['answers'].get(1)}, Q2={result['answers'].get(2)}, Q3={result['answers'].get(3)}")
        else:
            print(f"\n✗ Analysis failed: {result.get('error')}")

    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to server. Make sure Flask is running on port 5000")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    print("=" * 60)
    print("OMR Analysis API - Test Script")
    print("=" * 60 + "\n")

    if test_health():
        test_local_omr()
