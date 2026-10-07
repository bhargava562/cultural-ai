# Tamil Socio-Cultural Taxonomy & Dimensions

This reference taxonomy provides the foundational cultural grounding, linguistic registers, kinship structures, and UNESCO/Hofstede dimension mappings for the **Cultural Alignment Layer**.

---

## 1. Linguistic Varieties & Registers

| Variety / Dialect | Primary Regions | Characteristic Phonology & Markers | Social Context |
|---|---|---|---|
| **Standard Literary Tamil (செந்தமிழ்)** | Formal media, literature, official address | Classical morphology, non-elided suffixes (*-kiradhu*, *-udaiya*) | Formal speeches, ceremonial writings |
| **Colloquial Standard (பேச்சுத் தமிழ்)** | Urban TN, cinema, inter-regional conversation | Verb elisions (*vandhuten*, *pesuren*, *irukku*) | General conversational standard |
| **Chennai Tamil (மெட்ராஸ் பாஷை)** | Greater Chennai | Tanglish blending, Urdu/Telugu lexical influences (*baamaa*, *galatta*, *naina*, *kashtam*) | Youth, urban camaraderie, working-class solidarity |
| **Kongu Tamil (கொங்குத் தமிழ்)** | Coimbatore, Tiruppur, Erode, Salem | Invariable polite suffix *-unga*, distinct intonation, agrarian idioms | High domestic politeness, familial warmth |
| **Madurai / Southern Tamil** | Madurai, Theni, Dindigul, Tirunelveli | Emphatic suffixes (*-la*, *-ley*), distinctive greeting markers | Directness, passionate hospitality (*Virundhombal*) |
| **Tanglish (English-Tamil Code-Switch)** | Digital communication, urban youth, corporate tech | English syntax embedded with Tamil pragmatic particles (*pa*, *da*, *di*, *la*, *machi*) | Peer chats, modern technical discourse |

---

## 2. Honorific & Kinship Systems (*Maryadhai & Uravumurai*)

### Honorific Pronouns & Verb Conjugations
- **Informal / Intimate / Junior**: *Nee* (நீ) -> *Vaa* (வா), *Po* (போ), *Sol* (சொல்). Used only for peers, younger relatives, or intimate friends.
- **Formal / Elder / Respectful**: *Neenga* (நீங்கள்) -> *Vaanga* (வாருங்கள்), *Ponga* (போங்கள்), *Sollunga* (சொல்லுங்கள்). Mandatory for strangers, professionals, and all elders.

### Kinship Network (Key Distinctions)
- **Maternal vs. Paternal Aunts/Uncles**:
  - Father's elder brother: *Periyappa* (பெரியப்பா)
  - Father's younger brother: *Chithappa* (சித்தப்பா)
  - Mother's elder sister: *Periyamma* (பெரியம்மா)
  - Mother's younger sister: *Chithi* (சித்தி)
  - Maternal uncle / Father-in-law: *Mama* (மாமா)
  - Paternal aunt / Mother-in-law: *Athai* (அத்தை)
- **Cross-Cousin vs. Parallel Cousin Dynamics**: Culturally significant in traditional community events.

---

## 3. Cultural Pillars & Core Values

1. **Virundhombal (விருந்தோம்பல் - Hospitality)**:
   - Universal duty of welcoming guests with water, warm words, and generous food. Refusing to feed a visitor is a grave social transgression.
2. **Aram (அறம் - Moral Righteousness)**:
   - Rooted in classical Sangam literature and the *Thirukkural*. Righteous conduct, benevolence (*Eegai*), and community duty.
3. **Kudumbam & Valviyal (குடும்பம் - Familial Interdependence)**:
   - Deep intergenerational ties, respect for ancestors (*Munnor Vazhipaadu*), consultation with elders before major decisions.
4. **Festivals & Seasonal Rhythms**:
   - **Pongal (தைப்பொங்கல்)**: 4-day harvest celebration thanking the Sun (*Surya*), Cattle (*Mattu Pongal*), and nature. Key symbols: freshly harvested sugarcane, earthen pot boiling over (*Pongalo Pongal*), Kolam.
   - **Tamil New Year (புத்தாண்டு - Chithirai 1)**: Auspicious beginnings, *Manga Pachadi* (incorporating 6 tastes: sweet, sour, bitter, pungent, salty, astringent to symbolize life's varied experiences).
   - **Aadi Perukku**: Celebration of river Kaveri and water fertility.

---

## 4. Hofstede Cultural Dimension Mapping for Tamil Context

| Dimension | Score / Tendency | Manifestation in LLM Alignment |
|---|---|---|
| **Power Distance** | High | Respect for seniority, hierarchy, and status. LLM must not force overly egalitarian informal address (*first names*) on elders. |
| **Individualism vs. Collectivism** | Highly Collectivist | Decisions, celebrations, and ethical dilemmas involve extended family and community harmony rather than pure individual autonomy. |
| **Uncertainty Avoidance** | Moderate-High | Valuing ritual precision, established calendar dates (*Nalla Neram*, *Muhurtham*), and customary decorum. |
| **Long-Term Orientation** | High | Values perseverance (*Uzhaippu*), savings, intergenerational sacrifice, education (*Kalvi*). |

---

## 5. Anti-Bias & Sensitivity Guardrails

- **Caste Discrimination**: Strictly reject any caste-coded hierarchy, stereotypes about occupation or dietary preferences, or endogamy policing.
- **Linguistic Elitism**: Do not treat colloquial dialects or Tanglish as "degenerate" or "inferior" to classical Tamil. Validate everyday living speech.
- **Gender Stereotypes**: Avoid framing women solely in passive domestic roles or enforcing patriarchal strictures around festival duties.
