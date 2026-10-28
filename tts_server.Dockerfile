FROM tts_server-server:1.0

USER root

# European Portuguese voice (pt_tuga): the tts_eu_pt package; its weights are mounted under data/eu_pt (compose).
# 0.1.29 from ./wheels (built from ~/env/assets/tts_eu_pt; 0.1.29: the vowel after a joined linking z held 3 frames (João said "os manos"); 0.1.28: a linking z said as one word with the next vowel ("os humanos" uzuˈmɐnuʃ; Ana dragged it apart); 0.1.27: "especificamente" without the stray "_" (was heard "e aí especificamente"); 0.1.26: clock times keep the hour stressed, no pause before the hour, hour + "e" + minutes said as one word ("dezoit-ui-trinta"), "e meia" as minutes; 0.1.25: feminine participles read like their masculine (indexada with ks), "Euribór", João at pace 1.05; 0.1.24: João's preset for Kokoro's am_michael voicepack, pace 1.10; 0.1.23: repeatable audio, the same text always gives the same file; 0.1.22: two voices, Ana and João, at -16 LUFS, enclitic -o held, MBWAY; 0.1.21: a 50 ms gap after "treze"; 0.1.20: the z of "treze" held 100 ms; 0.1.19: "em" as ɐ̃j̃, "Áveiro", decimals said as one unit, "treze" with its last syllable kept; 0.1.18: pinned G2P stack, "dezoito" open as in Lisbon (dezóito); 0.1.17: the Cortex agent's 0.1.16 cases (Dr.ª, p.p., e-mails, DL n.º 74-A/2017, MB WAY), number words keep their final vowel at rate 1.08; 0.1.16: numbers and acronyms said as one unit, timing fixes at sentence ends and word ends, inline pronunciation overrides, rate 1.08 for the Parliament epoch-19 weights in data/eu_pt; earlier: plurals/feminines from their base, verb stem vowels, enclitics, ending and x rules measured on a pt-PT lexicon, written-accent stress of unknown words, stress of -ente/-ência words, -veis plurals, hyphenated identifiers, validator fixes (voicing, dates, acronyms, titles), "a bordo" said once, sentences read in context, dates/fractions/phone numbers,
# x sounds ("existe", "fixas"), nasal vowels, number gender, units/currencies, Roman numerals) until it is on PyPI;
# its dependencies come from PyPI as before.
COPY wheels/tts_eu_pt-0.1.29-py3-none-any.whl /tmp/
RUN pip install --no-cache-dir /tmp/tts_eu_pt-0.1.29-py3-none-any.whl && rm /tmp/tts_eu_pt-0.1.29-py3-none-any.whl

COPY *.py /tts_server

EXPOSE 7700

WORKDIR /tts_server

CMD ["python", "-u", "tts_server.py"]
