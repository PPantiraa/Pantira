import streamlit as st
import openai
import pandas as pd
from nltk.corpus import wordnet
from PyDictionary import PyDictionary

# Get the API key from the sidebar called OpenAI API key
user_api_key = st.sidebar.text_input("OpenAI API key", type="password")

# Set the OpenAI API key
openai.api_key = user_api_key

dictionary = PyDictionary()
prompt = """Act as an AI English teacher. You will receive Thai sentences
            and you have to translate Thai sentences to English. Then, 
            you have to identify interesting words to create a table with 
            four columns:
            - "Vocabulary" - the interesting word
            - "Definition" - the definition of the interesting word
            - "Synonym" - Synonym of the interesting word
            - "Antonym" - Antonym of the interesting word
        """    

st.title('Translation, Vocabulary, Definition, Synonym, and Antonym')
st.markdown('Input Thai sentences that you want to translate.')

user_input = st.text_area("Enter Thai sentences here:", "Your sentences here")

# translate button after text input
if st.button('Translate'):
    try:
        messages_so_far = [
            {"role": "system", "content": prompt},
            {'role': 'user', 'content': user_input},
        ]

        # Use openai.ChatCompletion.create instead of client.chat.completions.create
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages_so_far
        )

        translated_text = response['choices'][0]['message']['content']

        st.markdown('**AI response:**')
        st.write(translated_text)

        # Extract interesting words from the translated text
        words = translated_text.split()
        interesting_words = [word.strip('.,') for word in words if len(word) > 5] 

        # Create a table to display vocabulary, definition, synonym, and antonym
        table_data = {"Vocabulary": [], "Definition": [], "Synonym": [], "Antonym": []}
        for word in interesting_words:
            meanings = dictionary.meaning(word)
            if meanings:
                definition = ', '.join(item if isinstance(item, str) else ', '.join(item) for item in meanings.values())

                # Get synonym using NLTK
                synonyms = set()
                for syn in wordnet.synsets(word):
                    for lemma in syn.lemmas():
                        synonyms.add(lemma.name())
                synonyms.discard(word)  # Remove the word itself from synonym
                synonym_str = ', '.join(synonyms) if synonyms else '-'

                # Get antonym using NLTK
                antonyms = set()
                for syn in wordnet.synsets(word):
                    for lemma in syn.lemmas():
                        if lemma.antonyms():
                            antonyms.update(lemma.antonyms())
                antonym_str = ', '.join(antonym.name() for antonym in antonyms) if antonyms else '-'

                table_data["Vocabulary"].append(word)
                table_data["Definition"].append(definition)
                table_data["Synonym"].append(synonym_str)
                table_data["Antonym"].append(antonym_str)

        df = pd.DataFrame(table_data)
        st.write("### Interesting Words and Information")
        st.dataframe(df)

    except Exception as e:
        st.error(f"Error with OpenAI: {e}")
