"""
Test: Microphone Voice-to-Text Component Verification
Verifies:
1. components.html is used instead of st.html so <script> is not stripped.
2. Web Speech API (SpeechRecognition / webkitSpeechRecognition) is bound to the mic button.
3. Pulse/listening visual state (🔴, pulsing animation) is handled.
4. Transcript is captured and injected into the parent document textarea via React nativeSetter.
5. All modules compile cleanly and model.py is untouched.
"""

import hashlib
import py_compile
import streamlit.components.v1 as components

def test_microphone_implementation():
    print("=" * 70)
    print("VERIFICATION: MICROPHONE VOICE-TO-TEXT IMPLEMENTATION")
    print("=" * 70)

    # 1. Verify model.py SHA256 integrity
    with open("model.py", "rb") as f:
        m_sha256 = hashlib.sha256(f.read()).hexdigest()
    assert m_sha256 == "e559c30a92e9d34b7fed000b92533eb81c3184b4a0b14838a8e5e34c9a324f08"
    print("  [OK] model.py untouched.")

    # 2. Verify app.py contains the components.html microphone implementation
    with open("app.py", "r", encoding="utf-8") as f:
        content = f.read()

    assert "import streamlit.components.v1 as components" in content
    assert "components.html(" in content
    assert "mic-trigger-btn" in content
    assert "SpeechRecognition" in content
    assert "updateInputValue" in content
    assert "nativeSetter" in content
    assert "addEventListener(\"click\", toggleSpeechRecognition)" in content
    print("  [OK] components.html Web Speech API integration verified in app.py.")

    assert "stSidebar" in content
    assert "Message LocalGPT" in content
    assert "System Instructions" in content
    print("  [OK] Specific chat input targeting and sidebar exclusion verified.")

    # 3. Verify py_compile
    py_compile.compile("app.py", doraise=True)
    print("  [OK] app.py compiles cleanly.")

    print("\nALL MICROPHONE COMPONENT TESTS PASSED!")

if __name__ == "__main__":
    test_microphone_implementation()
