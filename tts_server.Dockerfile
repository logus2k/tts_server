FROM tts_server-server:1.0

USER root

# European Portuguese voices (pt_ana, pt_joao, pt_amelia, pt_rui): the tts_eu_pt package; its weights are mounted under
# data/eu_pt (compose). 0.1.33 from ./wheels (built from ~/env/assets/tts_eu_pt): 0.1.33, the listener's fixes of
# 2026-11-01 ("3 x 4" read "por", verb/noun homographs gósto/almóço, "treze" before a vowel, IBAN as "íban" and its
# digits); 0.1.32, the licence-clean model's four voice presets, no private-lexicon code, ordinals from unicode-rbnf.
# Earlier releases: the tts_eu_pt git history. Its dependencies come from PyPI.
COPY wheels/tts_eu_pt-0.1.33-py3-none-any.whl /tmp/
RUN pip install --no-cache-dir /tmp/tts_eu_pt-0.1.33-py3-none-any.whl && rm /tmp/tts_eu_pt-0.1.33-py3-none-any.whl

COPY *.py /tts_server

EXPOSE 7700

WORKDIR /tts_server

CMD ["python", "-u", "tts_server.py"]
