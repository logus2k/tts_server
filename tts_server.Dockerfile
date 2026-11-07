FROM tts_server-server:1.0

USER root

# European Portuguese voices (pt_ana, pt_joao, pt_amelia, pt_rui): the tts_eu_pt package; its weights are mounted under
# data/eu_pt (compose). 0.1.38 from ./wheels (built from ~/env/assets/tts_eu_pt): 0.1.38, the silence opening each chunk
# cut from the finished audio (0.1.35-0.1.37 shortened it in the model, which changed the words after it); 0.1.37, Ana louder (-14 LUFS target) to
# balance João in dialogues; 0.1.36, nasal a always ɐ̃ and -ãs plurals
# read from the singular ("ecrãs" eˈkɾɐ̃ʃ, was eˈkɾãʃ); 0.1.35, question intonation (opening
# raised, low on the last stress, rise to the end) and no 0.4-1.1 s silence before each chunk; 0.1.34, "amanhã" with an open first a
# and accented verb homographs read with the noun's syllables (alˈmɔsu, not ɐˈlmɔsu); 0.1.33, the listener's fixes of
# 2026-11-01 ("3 x 4" read "por", verb/noun homographs gósto/almóço, "treze" before a vowel, IBAN as "íban" and its
# digits); 0.1.32, the licence-clean model's four voice presets, no private-lexicon code, ordinals from unicode-rbnf.
# Earlier releases: the tts_eu_pt git history. Its dependencies come from PyPI.
COPY wheels/tts_eu_pt-0.1.38-py3-none-any.whl /tmp/
RUN pip install --no-cache-dir /tmp/tts_eu_pt-0.1.38-py3-none-any.whl && rm /tmp/tts_eu_pt-0.1.38-py3-none-any.whl

COPY *.py /tts_server

EXPOSE 7700

WORKDIR /tts_server

CMD ["python", "-u", "tts_server.py"]
