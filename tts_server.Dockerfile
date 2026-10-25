FROM tts_server-server:1.0

USER root

# European Portuguese voice (pt_tuga): the tts_eu_pt package; its weights are mounted under data/eu_pt (compose).
# 0.1.15 from ./wheels (built from ~/env/assets/tts_eu_pt: plurals/feminines from their base, verb stem vowels, enclitics, ending and x rules measured on a pt-PT lexicon, written-accent stress of unknown words, stress of -ente/-ência words, -veis plurals, hyphenated identifiers, validator fixes (voicing, dates, acronyms, titles), "a bordo" said once, sentences read in context, dates/fractions/phone numbers,
# x sounds ("existe", "fixas"), nasal vowels, number gender, units/currencies, Roman numerals) until it is on PyPI;
# its dependencies come from PyPI as before.
COPY wheels/tts_eu_pt-0.1.15-py3-none-any.whl /tmp/
RUN pip install --no-cache-dir /tmp/tts_eu_pt-0.1.15-py3-none-any.whl && rm /tmp/tts_eu_pt-0.1.15-py3-none-any.whl

COPY *.py /tts_server

EXPOSE 7700

WORKDIR /tts_server

CMD ["python", "-u", "tts_server.py"]
