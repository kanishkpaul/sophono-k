import tkinter as tk
from tkinter import simpledialog
from groq import Groq

def get_lyrics_from_user():
    root = tk.Tk()
    root.withdraw()
    lyrics = simpledialog.askstring("Lyrics Input", "Paste the song lyrics:")
    return lyrics

def analyze_lyrics(lyrics):
    if not lyrics or lyrics.strip() == "":
        return {"error": "No lyrics provided"}
    
    client = Groq(
        api_key="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",
    )
    
    prompt = f"""Analyze the following song lyrics and provide a detailed assessment:

    Lyrics:
    {lyrics}

    Your analysis should include:
    1. Emotional appeal and relatability
    2. Catchiness and memorability of phrases
    3. Repetition and hook effectiveness
    4. Current trends in popular music lyrics
    5. Potential audience reach
    6. Genre analysis (output in <genre></genre> tags)
    7. Comparison with popular songs in this genre
    8. Viral potential score (1-10) based on lyrics alone

    Format your response with clear sections for each analysis point.
    """
    
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-70b-8192",
            temperature=0.7,
            max_tokens=1024,
            stream=False,
        )
        
        analysis = response.choices[0].message.content
        return {"analysis": analysis}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    lyrics = get_lyrics_from_user()
    if lyrics:
        analysis = analyze_lyrics(lyrics)
        print("\n=== Lyrics Analysis ===")
        if "error" in analysis:
            print(analysis["error"])
        else:
            print(analysis["analysis"])