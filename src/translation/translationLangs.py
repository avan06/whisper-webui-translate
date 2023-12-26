from functools import cmp_to_key

class Lang():
    def __init__(self, code: str, *names: str):
        self.code = code
        self.names = names
        
    def __repr__(self):
        return f"code:{self.code}, name:{self.names}"

class TranslationLang():
    def __init__(self, code: str, name: str):
        self.nllb = Lang(code, name)
        self.whisper = None
        self.m2m100  = None
        self.seamlessT_Tx = None
        self.seamlessS_Sp = None
        
    def Whisper(self, code: str, *names: str):
        self.whisper = Lang(code, *names)
        if self.m2m100 is None:
            self.m2m100 = self.whisper
        return self
    
    def M2M100(self, code: str, name: str):
        self.m2m100 = Lang(code, name)
        return self
    
    def SeamlessT_Tx(self, code: str, name: str):
        self.seamlessT_Tx = Lang(code, name)
        if self.seamlessS_Sp is None:
            self.seamlessS_Sp = self.seamlessT_Tx
        return self
    
    def SeamlessS_Sp(self, code: str, name: str):
        self.seamlessS_Sp = Lang(code, name)
        return self

    def __repr__(self):
        result = ""
        if self.nllb:
            result += f"NLLB={self.nllb} "
        if self.whisper:
            result += f"WHISPER={self.whisper} "
        if self.m2m100:
            result += f"M2M100={self.m2m100} "
        if self.seamlessT_Tx:
            result += f"SeamlessTx={self.seamlessT_Tx} "
        if self.seamlessS_Sp:
            result += f"seamlessSp={self.seamlessS_Sp} "
        return f"Language {result}"
    
"""
Model available Languages

[NLLB]
ace_Latn:Acehnese (Latin script), aka_Latn:Akan, als_Latn:Tosk Albanian, amh_Ethi:Amharic, asm_Beng:Assamese, awa_Deva:Awadhi, ayr_Latn:Central Aymara, azb_Arab:South Azerbaijani, azj_Latn:North Azerbaijani, bak_Cyrl:Bashkir, bam_Latn:Bambara, ban_Latn:Balinese, bel_Cyrl:Belarusian, bem_Latn:Bemba, ben_Beng:Bengali, bho_Deva:Bhojpuri, bjn_Latn:Banjar (Latin script), bod_Tibt:Standard Tibetan, bug_Latn:Buginese, ceb_Latn:Cebuano, cjk_Latn:Chokwe, ckb_Arab:Central Kurdish, crh_Latn:Crimean Tatar, cym_Latn:Welsh, dik_Latn:Southwestern Dinka, diq_Latn:Southern Zaza, dyu_Latn:Dyula, dzo_Tibt:Dzongkha, ewe_Latn:Ewe, fao_Latn:Faroese, fij_Latn:Fijian, fon_Latn:Fon, fur_Latn:Friulian, fuv_Latn:Nigerian Fulfulde, gaz_Latn:West Central Oromo, gla_Latn:Scottish Gaelic, gle_Latn:Irish, grn_Latn:Guarani, guj_Gujr:Gujarati, hat_Latn:Haitian Creole, hau_Latn:Hausa, hin_Deva:Hindi, hne_Deva:Chhattisgarhi, hye_Armn:Armenian, ibo_Latn:Igbo, ilo_Latn:Ilocano, ind_Latn:Indonesian, jav_Latn:Javanese, kab_Latn:Kabyle, kac_Latn:Jingpho, kam_Latn:Kamba, kan_Knda:Kannada, kas_Arab:Kashmiri (Arabic script), kas_Deva:Kashmiri (Devanagari script), kat_Geor:Georgian, kaz_Cyrl:Kazakh, kbp_Latn:Kabiyè, kea_Latn:Kabuverdianu, khk_Cyrl:Halh Mongolian, khm_Khmr:Khmer, kik_Latn:Kikuyu, kin_Latn:Kinyarwanda, kir_Cyrl:Kyrgyz, kmb_Latn:Kimbundu, kmr_Latn:Northern Kurdish, knc_Arab:Central Kanuri (Arabic script), knc_Latn:Central Kanuri (Latin script), kon_Latn:Kikongo, lao_Laoo:Lao, lij_Latn:Ligurian, lim_Latn:Limburgish, lin_Latn:Lingala, lmo_Latn:Lombard, ltg_Latn:Latgalian, ltz_Latn:Luxembourgish, lua_Latn:Luba-Kasai, lug_Latn:Ganda, luo_Latn:Luo, lus_Latn:Mizo, mag_Deva:Magahi, mai_Deva:Maithili, mal_Mlym:Malayalam, mar_Deva:Marathi, min_Latn:Minangkabau (Latin script), mlt_Latn:Maltese, mni_Beng:Meitei (Bengali script), mos_Latn:Mossi, mri_Latn:Maori, mya_Mymr:Burmese, npi_Deva:Nepali, nso_Latn:Northern Sotho, nus_Latn:Nuer, nya_Latn:Nyanja, ory_Orya:Odia, pag_Latn:Pangasinan, pan_Guru:Eastern Panjabi, pap_Latn:Papiamento, pbt_Arab:Southern Pashto, pes_Arab:Western Persian, plt_Latn:Plateau Malagasy, prs_Arab:Dari, quy_Latn:Ayacucho Quechua, run_Latn:Rundi, sag_Latn:Sango, san_Deva:Sanskrit, sat_Beng:Santali, scn_Latn:Sicilian, shn_Mymr:Shan, sin_Sinh:Sinhala, smo_Latn:Samoan, sna_Latn:Shona, snd_Arab:Sindhi, som_Latn:Somali, sot_Latn:Southern Sotho, srd_Latn:Sardinian, ssw_Latn:Swati, sun_Latn:Sundanese, swh_Latn:Swahili, szl_Latn:Silesian, tam_Taml:Tamil, taq_Latn:Tamasheq (Latin script), tat_Cyrl:Tatar, tel_Telu:Telugu, tgk_Cyrl:Tajik, tgl_Latn:Tagalog, tha_Thai:Thai, tir_Ethi:Tigrinya, tpi_Latn:Tok Pisin, tsn_Latn:Tswana, tso_Latn:Tsonga, tuk_Latn:Turkmen, tum_Latn:Tumbuka, tur_Latn:Turkish, twi_Latn:Twi, tzm_Tfng:Central Atlas Tamazight, uig_Arab:Uyghur, umb_Latn:Umbundu, urd_Arab:Urdu, uzn_Latn:Northern Uzbek, vec_Latn:Venetian, war_Latn:Waray, wol_Latn:Wolof, xho_Latn:Xhosa, ydd_Hebr:Eastern Yiddish, yor_Latn:Yoruba, zsm_Latn:Standard Malay, zul_Latn:Zulu
https://github.com/facebookresearch/LASER/blob/main/nllb/README.md

In the NLLB model, languages are identified by a FLORES-200 code of the form {language}_{script}, where the language is an ISO 639-3 code and the script is an ISO 15924 code.
https://github.com/sillsdev/serval/wiki/FLORES%E2%80%90200-Language-Code-Resolution-for-NMT-Engine

[whisper]
en:english, zh:chinese, de:german, es:spanish, ru:russian, ko:korean, fr:french, ja:japanese, pt:portuguese, tr:turkish, pl:polish, ca:catalan, nl:dutch, ar:arabic, sv:swedish, it:italian, id:indonesian, hi:hindi, fi:finnish, vi:vietnamese, he:hebrew, uk:ukrainian, el:greek, ms:malay, cs:czech, ro:romanian, da:danish, hu:hungarian, ta:tamil, no:norwegian, th:thai, ur:urdu, hr:croatian, bg:bulgarian, lt:lithuanian, la:latin, mi:maori, ml:malayalam, cy:welsh, sk:slovak, te:telugu, fa:persian, lv:latvian, bn:bengali, sr:serbian, az:azerbaijani, sl:slovenian, kn:kannada, et:estonian, mk:macedonian, br:breton, eu:basque, is:icelandic, hy:armenian, ne:nepali, mn:mongolian, bs:bosnian, kk:kazakh, sq:albanian, sw:swahili, gl:galician, mr:marathi, pa:punjabi, si:sinhala, km:khmer, sn:shona, yo:yoruba, so:somali, af:afrikaans, oc:occitan, ka:georgian, be:belarusian, tg:tajik, sd:sindhi, gu:gujarati, am:amharic, yi:yiddish, lo:lao, uz:uzbek, fo:faroese, ht:haitian creole, ps:pashto, tk:turkmen, nn:nynorsk, mt:maltese, sa:sanskrit, lb:luxembourgish, my:myanmar, bo:tibetan, tl:tagalog, mg:malagasy, as:assamese, tt:tatar, haw:hawaiian, ln:lingala, ha:hausa, ba:bashkir, jw:javanese, su:sundanese, yue:cantonese,
https://github.com/openai/whisper/blob/main/whisper/tokenizer.py

[m2m100]
af:Afrikaans, am:Amharic, ar:Arabic, ast:Asturian, az:Azerbaijani, ba:Bashkir, be:Belarusian, bg:Bulgarian, bn:Bengali, br:Breton, bs:Bosnian, ca:Catalan; Valencian, ceb:Cebuano, cs:Czech, cy:Welsh, da:Danish, de:German, el:Greek, en:English, es:Spanish, et:Estonian, fa:Persian, ff:Fulah, fi:Finnish, fr:French, fy:Western Frisian, ga:Irish, gd:Gaelic; Scottish Gaelic, gl:Galician, gu:Gujarati, ha:Hausa, he:Hebrew, hi:Hindi, hr:Croatian, ht:Haitian; Haitian Creole, hu:Hungarian, hy:Armenian, id:Indonesian, ig:Igbo, ilo:Iloko, is:Icelandic, it:Italian, ja:Japanese, jv:Javanese, ka:Georgian, kk:Kazakh, km:Central Khmer, kn:Kannada, ko:Korean, lb:Luxembourgish; Letzeburgesch, lg:Ganda, ln:Lingala, lo:Lao, lt:Lithuanian, lv:Latvian, mg:Malagasy, mk:Macedonian, ml:Malayalam, mn:Mongolian, mr:Marathi, ms:Malay, my:Burmese, ne:Nepali, nl:Dutch; Flemish, no:Norwegian, ns:Northern Sotho, Occitan (oc:post 1500), or:Oriya, pa:Panjabi; Punjabi, pl:Polish, ps:Pushto; Pashto, pt:Portuguese, ro:Romanian; Moldavian; Moldovan, ru:Russian, sd:Sindhi, si:Sinhala; Sinhalese, sk:Slovak, sl:Slovenian, so:Somali, sq:Albanian, sr:Serbian, ss:Swati, su:Sundanese, sv:Swedish, sw:Swahili, ta:Tamil, th:Thai, tl:Tagalog, tn:Tswana, tr:Turkish, uk:Ukrainian, ur:Urdu, uz:Uzbek, vi:Vietnamese, wo:Wolof, xh:Xhosa, yi:Yiddish, yo:Yoruba, zh:Chinese, zu:Zulu
https://huggingface.co/facebook/m2m100_1.2B

The available languages for m2m100 and whisper are almost identical. Most of the codes correspond to the ISO 639-1 standard. For detailed information, please refer to the official documentation provided.

[seamless]
Source:Sp
afr:Afrikaans, amh:Amharic, arb:Modern Standard Arabic, ary:Moroccan Arabic, arz:Egyptian Arabic, asm:Assamese, ast:Asturian, azj:North Azerbaijani, bel:Belarusian, ben:Bengali, bos:Bosnian, bul:Bulgarian, cat:Catalan, ceb:Cebuano, ces:Czech, ckb:Central Kurdish, cmn:Mandarin Chinese, cmn_Hant:Mandarin Chinese, cym:Welsh, dan:Danish, deu:German, ell:Greek, eng:English, est:Estonian, eus:Basque, fin:Finnish, fra:French, fuv:Nigerian Fulfulde, gaz:West Central Oromo, gle:Irish, glg:Galician, guj:Gujarati, heb:Hebrew, hin:Hindi, hrv:Croatian, hun:Hungarian, hye:Armenian, ibo:Igbo, ind:Indonesian, isl:Icelandic, ita:Italian, jav:Javanese, jpn:Japanese, kam:Kamba, kan:Kannada, kat:Georgian, kaz:Kazakh, kea:Kabuverdianu, khk:Halh Mongolian, khm:Khmer, kir:Kyrgyz, kor:Korean, lao:Lao, lit:Lithuanian, ltz:Luxembourgish, lug:Ganda, luo:Luo, lvs:Standard Latvian, mai:Maithili, mal:Malayalam, mar:Marathi, mkd:Macedonian, mlt:Maltese, mni:Meitei, mya:Burmese, nld:Dutch, nno:Norwegian Nynorsk, nob:Norwegian Bokmål, npi:Nepali, nya:Nyanja, oci:Occitan, ory:Odia, pan:Punjabi, pbt:Southern Pashto, pes:Western Persian, pol:Polish, por:Portuguese, ron:Romanian, rus:Russian, slk:Slovak, slv:Slovenian, sna:Shona, snd:Sindhi, som:Somali, spa:Spanish, srp:Serbian, swe:Swedish, swh:Swahili, tam:Tamil, tel:Telugu, tgk:Tajik, tgl:Tagalog, tha:Thai, tur:Turkish, ukr:Ukrainian, urd:Urdu, uzn:Northern Uzbek, vie:Vietnamese, xho:Xhosa, yor:Yoruba, yue:Cantonese, zlm:Colloquial Malay, zul:Zulu, 
Target:Tx
afr:Afrikaans, amh:Amharic, arb:Modern Standard Arabic, ary:Moroccan Arabic, arz:Egyptian Arabic, asm:Assamese, azj:North Azerbaijani, bel:Belarusian, ben:Bengali, bos:Bosnian, bul:Bulgarian, cat:Catalan, ceb:Cebuano, ces:Czech, ckb:Central Kurdish, cmn:Mandarin Chinese, cmn_Hant:Mandarin Chinese, cym:Welsh, dan:Danish, deu:German, ell:Greek, eng:English, est:Estonian, eus:Basque, fin:Finnish, fra:French, fuv:Nigerian Fulfulde, gaz:West Central Oromo, gle:Irish, glg:Galician, guj:Gujarati, heb:Hebrew, hin:Hindi, hrv:Croatian, hun:Hungarian, hye:Armenian, ibo:Igbo, ind:Indonesian, isl:Icelandic, ita:Italian, jav:Javanese, jpn:Japanese, kan:Kannada, kat:Georgian, kaz:Kazakh, khk:Halh Mongolian, khm:Khmer, kir:Kyrgyz, kor:Korean, lao:Lao, lit:Lithuanian, lug:Ganda, luo:Luo, lvs:Standard Latvian, mai:Maithili, mal:Malayalam, mar:Marathi, mkd:Macedonian, mlt:Maltese, mni:Meitei, mya:Burmese, nld:Dutch, nno:Norwegian Nynorsk, nob:Norwegian Bokmål, npi:Nepali, nya:Nyanja, ory:Odia, pan:Punjabi, pbt:Southern Pashto, pes:Western Persian, pol:Polish, por:Portuguese, ron:Romanian, rus:Russian, slk:Slovak, slv:Slovenian, sna:Shona, snd:Sindhi, som:Somali, spa:Spanish, srp:Serbian, swe:Swedish, swh:Swahili, tam:Tamil, tel:Telugu, tgk:Tajik, tgl:Tagalog, tha:Thai, tur:Turkish, ukr:Ukrainian, urd:Urdu, uzn:Northern Uzbek, vie:Vietnamese, yor:Yoruba, yue:Cantonese, zsm:Standard Malay(Source Tx only), zul:Zulu, 
Source:Sp, Tx ／ Target:Sp, Tx
arb:Modern Standard Arabic, ben:Bengali, cat:Catalan, ces:Czech, cmn:Mandarin Chinese, cmn_Hant:Mandarin Chinese, cym:Welsh, dan:Danish, deu:German, eng:English, est:Estonian, fin:Finnish, fra:French, hin:Hindi, ind:Indonesian, ita:Italian, jpn:Japanese, kor:Korean, mlt:Maltese, nld:Dutch, pes:Western Persian, pol:Polish, por:Portuguese, ron:Romanian, rus:Russian, slk:Slovak, spa:Spanish, swe:Swedish, swh:Swahili, tel:Telugu, tgl:Tagalog, tha:Thai, tur:Turkish, ukr:Ukrainian, urd:Urdu, uzn:Northern Uzbek, vie:Vietnamese, 
https://huggingface.co/facebook/seamless-m4t-v2-large
"""
TranslationLangs = [
    TranslationLang("ace_Arab", "Acehnese (Arabic script)"),
    TranslationLang("ace_Latn", "Acehnese (Latin script)"),
    TranslationLang("acm_Arab", "Mesopotamian Arabic").Whisper("ar", "Arabic"),
    TranslationLang("acq_Arab", "Ta’izzi-Adeni Arabic").Whisper("ar", "Arabic"),
    TranslationLang("aeb_Arab", "Tunisian Arabic"),
    TranslationLang("afr_Latn", "Afrikaans").Whisper("af", "Afrikaans").SeamlessT_Tx("afr", "Afrikaans"),
    TranslationLang("ajp_Arab", "South Levantine Arabic").Whisper("ar", "Arabic"),
    TranslationLang("aka_Latn", "Akan"),
    TranslationLang("amh_Ethi", "Amharic").Whisper("am", "Amharic").SeamlessT_Tx("amh", "Amharic"),
    TranslationLang("apc_Arab", "North Levantine Arabic").Whisper("ar", "Arabic"),
    TranslationLang("arb_Arab", "Modern Standard Arabic").Whisper("ar", "Arabic").SeamlessT_Tx("arb", "Modern Standard Arabic"),
    TranslationLang("arb_Latn", "Modern Standard Arabic (Romanized)"),
    TranslationLang("ars_Arab", "Najdi Arabic").Whisper("ar", "Arabic"),
    TranslationLang("ary_Arab", "Moroccan Arabic").Whisper("ar", "Arabic").SeamlessT_Tx("ary", "Moroccan Arabic"),
    TranslationLang("arz_Arab", "Egyptian Arabic").Whisper("ar", "Arabic").SeamlessT_Tx("arz", "Egyptian Arabic"),
    TranslationLang("asm_Beng", "Assamese").Whisper("as", "Assamese").SeamlessT_Tx("asm", "Assamese"),
    TranslationLang("ast_Latn", "Asturian").M2M100("ast", "Asturian").SeamlessS_Sp("ast", "Asturian"),
    TranslationLang("awa_Deva", "Awadhi"),
    TranslationLang("ayr_Latn", "Central Aymara"),
    TranslationLang("azb_Arab", "South Azerbaijani").Whisper("az", "Azerbaijani"),
    TranslationLang("azj_Latn", "North Azerbaijani").Whisper("az", "Azerbaijani").SeamlessT_Tx("azj", "North Azerbaijani"),
    TranslationLang("bak_Cyrl", "Bashkir").Whisper("ba", "Bashkir"),
    TranslationLang("bam_Latn", "Bambara"),
    TranslationLang("ban_Latn", "Balinese"),
    TranslationLang("bel_Cyrl", "Belarusian").Whisper("be", "Belarusian").SeamlessT_Tx("bel", "Belarusian"),
    TranslationLang("bem_Latn", "Bemba"),
    TranslationLang("ben_Beng", "Bengali").Whisper("bn", "Bengali").SeamlessT_Tx("ben", "Bengali"),
    TranslationLang("bho_Deva", "Bhojpuri"),
    TranslationLang("bjn_Arab", "Banjar (Arabic script)"),
    TranslationLang("bjn_Latn", "Banjar (Latin script)"),
    TranslationLang("bod_Tibt", "Standard Tibetan").Whisper("bo", "Tibetan"),
    TranslationLang("bos_Latn", "Bosnian").Whisper("bs", "Bosnian").SeamlessT_Tx("bos", "Bosnian"),
    TranslationLang("bug_Latn", "Buginese"),
    TranslationLang("bul_Cyrl", "Bulgarian").Whisper("bg", "Bulgarian").SeamlessT_Tx("bul", "Bulgarian"),
    TranslationLang("cat_Latn", "Catalan").Whisper("ca", "Catalan", "valencian").SeamlessT_Tx("cat", "Catalan"),
    TranslationLang("ceb_Latn", "Cebuano").M2M100("ceb", "Cebuano").SeamlessT_Tx("ceb", "Cebuano"),
    TranslationLang("ces_Latn", "Czech").Whisper("cs", "Czech").SeamlessT_Tx("ces", "Czech"),
    TranslationLang("cjk_Latn", "Chokwe"),
    TranslationLang("ckb_Arab", "Central Kurdish").SeamlessT_Tx("ckb", "Central Kurdish"),
    TranslationLang("crh_Latn", "Crimean Tatar"),
    TranslationLang("cym_Latn", "Welsh").Whisper("cy", "Welsh").SeamlessT_Tx("cym", "Welsh"),
    TranslationLang("dan_Latn", "Danish").Whisper("da", "Danish").SeamlessT_Tx("dan", "Danish"),
    TranslationLang("deu_Latn", "German").Whisper("de", "German").SeamlessT_Tx("deu", "German"),
    TranslationLang("dik_Latn", "Southwestern Dinka"),
    TranslationLang("dyu_Latn", "Dyula"),
    TranslationLang("dzo_Tibt", "Dzongkha"),
    TranslationLang("ell_Grek", "Greek").Whisper("el", "Greek").SeamlessT_Tx("ell", "Greek"),
    TranslationLang("eng_Latn", "English").Whisper("en", "English").SeamlessT_Tx("eng", "English"),
    TranslationLang("epo_Latn", "Esperanto"),
    TranslationLang("est_Latn", "Estonian").Whisper("et", "Estonian").SeamlessT_Tx("est", "Estonian"),
    TranslationLang("eus_Latn", "Basque").Whisper("eu", "Basque").SeamlessT_Tx("eus", "Basque"),
    TranslationLang("ewe_Latn", "Ewe"),
    TranslationLang("fao_Latn", "Faroese").Whisper("fo", "Faroese"),
    TranslationLang("fij_Latn", "Fijian"),
    TranslationLang("fin_Latn", "Finnish").Whisper("fi", "Finnish").SeamlessT_Tx("fin", "Finnish"),
    TranslationLang("fon_Latn", "Fon"),
    TranslationLang("fra_Latn", "French").Whisper("fr", "French").SeamlessT_Tx("fra", "French"),
    TranslationLang("fur_Latn", "Friulian"),
    TranslationLang("fuv_Latn", "Nigerian Fulfulde").M2M100("ff", "Fulah").SeamlessT_Tx("fuv", "Nigerian Fulfulde"),
    TranslationLang("gla_Latn", "Scottish Gaelic").M2M100("gd", "Scottish Gaelic"),
    TranslationLang("gle_Latn", "Irish").M2M100("ga", "Irish").SeamlessT_Tx("gle", "Irish"),
    TranslationLang("glg_Latn", "Galician").Whisper("gl", "Galician").SeamlessT_Tx("glg", "Galician"),
    TranslationLang("grn_Latn", "Guarani"),
    TranslationLang("guj_Gujr", "Gujarati").Whisper("gu", "Gujarati").SeamlessT_Tx("guj", "Gujarati"),
    TranslationLang("hat_Latn", "Haitian Creole").Whisper("ht", "Haitian creole", "haitian"),
    TranslationLang("hau_Latn", "Hausa").Whisper("ha", "Hausa"),
    TranslationLang("heb_Hebr", "Hebrew").Whisper("he", "Hebrew").SeamlessT_Tx("heb", "Hebrew"),
    TranslationLang("hin_Deva", "Hindi").Whisper("hi", "Hindi").SeamlessT_Tx("hin", "Hindi"),
    TranslationLang("hne_Deva", "Chhattisgarhi"),
    TranslationLang("hrv_Latn", "Croatian").Whisper("hr", "Croatian").SeamlessT_Tx("hrv", "Croatian"),
    TranslationLang("hun_Latn", "Hungarian").Whisper("hu", "Hungarian").SeamlessT_Tx("hun", "Hungarian"),
    TranslationLang("hye_Armn", "Armenian").Whisper("hy", "Armenian").SeamlessT_Tx("hye", "Armenian"),
    TranslationLang("ibo_Latn", "Igbo").M2M100("ig", "Igbo").SeamlessT_Tx("ibo", "Igbo"),
    TranslationLang("ilo_Latn", "Ilocano").M2M100("ilo", "Iloko"),
    TranslationLang("ind_Latn", "Indonesian").Whisper("id", "Indonesian").SeamlessT_Tx("ind", "Indonesian"),
    TranslationLang("isl_Latn", "Icelandic").Whisper("is", "Icelandic").SeamlessT_Tx("isl", "Icelandic"),
    TranslationLang("ita_Latn", "Italian").Whisper("it", "Italian").SeamlessT_Tx("ita", "Italian"),
    TranslationLang("jav_Latn", "Javanese").Whisper("jw", "Javanese").M2M100("jv", "Javanese").SeamlessT_Tx("jav", "Javanese"),
    TranslationLang("jpn_Jpan", "Japanese").Whisper("ja", "Japanese").SeamlessT_Tx("jpn", "Japanese"),
    TranslationLang("kab_Latn", "Kabyle"),
    TranslationLang("kac_Latn", "Jingpho"),
    TranslationLang("kam_Latn", "Kamba").SeamlessS_Sp("kam", "Kamba"),
    TranslationLang("kan_Knda", "Kannada").Whisper("kn", "Kannada").SeamlessT_Tx("kan", "Kannada"),
    TranslationLang("kas_Arab", "Kashmiri (Arabic script)"),
    TranslationLang("kas_Deva", "Kashmiri (Devanagari script)"),
    TranslationLang("kat_Geor", "Georgian").Whisper("ka", "Georgian").SeamlessT_Tx("kat", "Georgian"),
    TranslationLang("knc_Arab", "Central Kanuri (Arabic script)"),
    TranslationLang("knc_Latn", "Central Kanuri (Latin script)"),
    TranslationLang("kaz_Cyrl", "Kazakh").Whisper("kk", "Kazakh").SeamlessT_Tx("kaz", "Kazakh"),
    TranslationLang("kbp_Latn", "Kabiyè"),
    TranslationLang("kea_Latn", "Kabuverdianu").SeamlessS_Sp("kea", "Kabuverdianu"),
    TranslationLang("khm_Khmr", "Khmer").Whisper("km", "Khmer").SeamlessT_Tx("khm", "Khmer"),
    TranslationLang("kik_Latn", "Kikuyu"),
    TranslationLang("kin_Latn", "Kinyarwanda"),
    TranslationLang("kir_Cyrl", "Kyrgyz").SeamlessT_Tx("kir", "Kyrgyz"),
    TranslationLang("kmb_Latn", "Kimbundu"),
    TranslationLang("kmr_Latn", "Northern Kurdish"),
    TranslationLang("kon_Latn", "Kikongo"),
    TranslationLang("kor_Hang", "Korean").Whisper("ko", "Korean").SeamlessT_Tx("kor", "Korean"),
    TranslationLang("lao_Laoo", "Lao").Whisper("lo", "Lao").SeamlessT_Tx("lao", "Lao"),
    TranslationLang("lij_Latn", "Ligurian"),
    TranslationLang("lim_Latn", "Limburgish"),
    TranslationLang("lin_Latn", "Lingala").Whisper("ln", "Lingala"),
    TranslationLang("lit_Latn", "Lithuanian").Whisper("lt", "Lithuanian").SeamlessT_Tx("lit", "Lithuanian"),
    TranslationLang("lmo_Latn", "Lombard"),
    TranslationLang("ltg_Latn", "Latgalian"),
    TranslationLang("ltz_Latn", "Luxembourgish").Whisper("lb", "Luxembourgish", "letzeburgesch").SeamlessS_Sp("ltz", "Luxembourgish"),
    TranslationLang("lua_Latn", "Luba-Kasai"),
    TranslationLang("lug_Latn", "Ganda").M2M100("lg", "Ganda").SeamlessT_Tx("lug", "Ganda"),
    TranslationLang("luo_Latn", "Luo").SeamlessT_Tx("luo", "Luo"),
    TranslationLang("lus_Latn", "Mizo"),
    TranslationLang("lvs_Latn", "Standard Latvian").Whisper("lv", "Latvian").SeamlessT_Tx("lvs", "Standard Latvian"),
    TranslationLang("mag_Deva", "Magahi"),
    TranslationLang("mai_Deva", "Maithili").SeamlessT_Tx("mai", "Maithili"),
    TranslationLang("mal_Mlym", "Malayalam").Whisper("ml", "Malayalam").SeamlessT_Tx("mal", "Malayalam"),
    TranslationLang("mar_Deva", "Marathi").Whisper("mr", "Marathi").SeamlessT_Tx("mar", "Marathi"),
    TranslationLang("min_Arab", "Minangkabau (Arabic script)"),
    TranslationLang("min_Latn", "Minangkabau (Latin script)"),
    TranslationLang("mkd_Cyrl", "Macedonian").Whisper("mk", "Macedonian").SeamlessT_Tx("mkd", "Macedonian"),
    TranslationLang("plt_Latn", "Plateau Malagasy").Whisper("mg", "Malagasy"),
    TranslationLang("mlt_Latn", "Maltese").Whisper("mt", "Maltese").SeamlessT_Tx("mlt", "Maltese"),
    TranslationLang("mni_Beng", "Meitei (Bengali script)").SeamlessT_Tx("mni", "Meitei"),
    TranslationLang("khk_Cyrl", "Halh Mongolian").Whisper("mn", "Mongolian").SeamlessT_Tx("khk", "Halh Mongolian"),
    TranslationLang("mos_Latn", "Mossi"),
    TranslationLang("mri_Latn", "Maori").Whisper("mi", "Maori"),
    TranslationLang("mya_Mymr", "Burmese").Whisper("my", "Myanmar", "burmese").SeamlessT_Tx("mya", "Burmese"),
    TranslationLang("nld_Latn", "Dutch").Whisper("nl", "Dutch", "flemish").SeamlessT_Tx("nld", "Dutch"),
    TranslationLang("nno_Latn", "Norwegian Nynorsk").Whisper("nn", "Nynorsk").SeamlessT_Tx("nno", "Norwegian Nynorsk"),
    TranslationLang("nob_Latn", "Norwegian Bokmål").Whisper("no", "Norwegian").SeamlessT_Tx("nob", "Norwegian Bokmål"),
    TranslationLang("npi_Deva", "Nepali").Whisper("ne", "Nepali").SeamlessT_Tx("npi", "Nepali"),
    TranslationLang("nso_Latn", "Northern Sotho").M2M100("ns", "Northern Sotho"),
    TranslationLang("nus_Latn", "Nuer"),
    TranslationLang("nya_Latn", "Nyanja").SeamlessT_Tx("nya", "Nyanja"),
    TranslationLang("oci_Latn", "Occitan").Whisper("oc", "Occitan").SeamlessS_Sp("oci", "Occitan"),
    TranslationLang("gaz_Latn", "West Central Oromo").SeamlessT_Tx("gaz", "West Central Oromo"),
    TranslationLang("ory_Orya", "Odia").M2M100("or", "Oriya").SeamlessT_Tx("ory", "Odia"),
    TranslationLang("pag_Latn", "Pangasinan"),
    TranslationLang("pan_Guru", "Eastern Panjabi").Whisper("pa", "Punjabi", "panjabi").SeamlessT_Tx("pan", "Punjabi"),
    TranslationLang("pap_Latn", "Papiamento"),
    TranslationLang("pes_Arab", "Western Persian").Whisper("fa", "Persian").SeamlessT_Tx("pes", "Western Persian"),
    TranslationLang("pol_Latn", "Polish").Whisper("pl", "Polish").SeamlessT_Tx("pol", "Polish"),
    TranslationLang("por_Latn", "Portuguese").Whisper("pt", "Portuguese").SeamlessT_Tx("por", "Portuguese"),
    TranslationLang("prs_Arab", "Dari"),
    TranslationLang("pbt_Arab", "Southern Pashto").Whisper("ps", "Pashto", "pushto").SeamlessT_Tx("pbt", "Southern Pashto"),
    TranslationLang("quy_Latn", "Ayacucho Quechua"),
    TranslationLang("ron_Latn", "Romanian").Whisper("ro", "Romanian", "moldavian", "moldovan").SeamlessT_Tx("ron", "Romanian"),
    TranslationLang("run_Latn", "Rundi"),
    TranslationLang("rus_Cyrl", "Russian").Whisper("ru", "Russian").SeamlessT_Tx("rus", "Russian"),
    TranslationLang("sag_Latn", "Sango"),
    TranslationLang("san_Deva", "Sanskrit").Whisper("sa", "Sanskrit"),
    TranslationLang("sat_Olck", "Santali"),
    TranslationLang("scn_Latn", "Sicilian"),
    TranslationLang("shn_Mymr", "Shan"),
    TranslationLang("sin_Sinh", "Sinhala").Whisper("si", "Sinhala", "sinhalese"),
    TranslationLang("slk_Latn", "Slovak").Whisper("sk", "Slovak").SeamlessT_Tx("slk", "Slovak"),
    TranslationLang("slv_Latn", "Slovenian").Whisper("sl", "Slovenian").SeamlessT_Tx("slv", "Slovenian"),
    TranslationLang("smo_Latn", "Samoan"),
    TranslationLang("sna_Latn", "Shona").Whisper("sn", "Shona").SeamlessT_Tx("sna", "Shona"),
    TranslationLang("snd_Arab", "Sindhi").Whisper("sd", "Sindhi").SeamlessT_Tx("snd", "Sindhi"),
    TranslationLang("som_Latn", "Somali").Whisper("so", "Somali").SeamlessT_Tx("som", "Somali"),
    TranslationLang("sot_Latn", "Southern Sotho"),
    TranslationLang("spa_Latn", "Spanish").Whisper("es", "Spanish", "castilian").SeamlessT_Tx("spa", "Spanish"),
    TranslationLang("als_Latn", "Tosk Albanian").Whisper("sq", "Albanian"),
    TranslationLang("srd_Latn", "Sardinian"),
    TranslationLang("srp_Cyrl", "Serbian").Whisper("sr", "Serbian").SeamlessT_Tx("srp", "Serbian"),
    TranslationLang("ssw_Latn", "Swati").M2M100("ss", "Swati"),
    TranslationLang("sun_Latn", "Sundanese").Whisper("su", "Sundanese"),
    TranslationLang("swe_Latn", "Swedish").Whisper("sv", "Swedish").SeamlessT_Tx("swe", "Swedish"),
    TranslationLang("swh_Latn", "Swahili").Whisper("sw", "Swahili").SeamlessT_Tx("swh", "Swahili"),
    TranslationLang("szl_Latn", "Silesian"),
    TranslationLang("tam_Taml", "Tamil").Whisper("ta", "Tamil").SeamlessT_Tx("tam", "Tamil"),
    TranslationLang("tat_Cyrl", "Tatar").Whisper("tt", "Tatar"),
    TranslationLang("tel_Telu", "Telugu").Whisper("te", "Telugu").SeamlessT_Tx("tel", "Telugu"),
    TranslationLang("tgk_Cyrl", "Tajik").Whisper("tg", "Tajik").SeamlessT_Tx("tgk", "Tajik"),
    TranslationLang("tgl_Latn", "Tagalog").Whisper("tl", "Tagalog").SeamlessT_Tx("tgl", "Tagalog"),
    TranslationLang("tha_Thai", "Thai").Whisper("th", "Thai").SeamlessT_Tx("tha", "Thai"),
    TranslationLang("tir_Ethi", "Tigrinya"),
    TranslationLang("taq_Latn", "Tamasheq (Latin script)"),
    TranslationLang("taq_Tfng", "Tamasheq (Tifinagh script)"),
    TranslationLang("tpi_Latn", "Tok Pisin"),
    TranslationLang("tsn_Latn", "Tswana").M2M100("tn", "Tswana"),
    TranslationLang("tso_Latn", "Tsonga"),
    TranslationLang("tuk_Latn", "Turkmen").Whisper("tk", "Turkmen"),
    TranslationLang("tum_Latn", "Tumbuka"),
    TranslationLang("tur_Latn", "Turkish").Whisper("tr", "Turkish").SeamlessT_Tx("tur", "Turkish"),
    TranslationLang("twi_Latn", "Twi"),
    TranslationLang("tzm_Tfng", "Central Atlas Tamazight"),
    TranslationLang("uig_Arab", "Uyghur"),
    TranslationLang("ukr_Cyrl", "Ukrainian").Whisper("uk", "Ukrainian").SeamlessT_Tx("ukr", "Ukrainian"),
    TranslationLang("umb_Latn", "Umbundu"),
    TranslationLang("urd_Arab", "Urdu").Whisper("ur", "Urdu").SeamlessT_Tx("urd", "Urdu"),
    TranslationLang("uzn_Latn", "Northern Uzbek").Whisper("uz", "Uzbek").SeamlessT_Tx("uzn", "Northern Uzbek"),
    TranslationLang("vec_Latn", "Venetian"),
    TranslationLang("vie_Latn", "Vietnamese").Whisper("vi", "Vietnamese").SeamlessT_Tx("vie", "Vietnamese"),
    TranslationLang("war_Latn", "Waray"),
    TranslationLang("wol_Latn", "Wolof").M2M100("wo", "Wolof"),
    TranslationLang("xho_Latn", "Xhosa").M2M100("xh", "Xhosa").SeamlessS_Sp("xho", "Xhosa"),
    TranslationLang("ydd_Hebr", "Eastern Yiddish").Whisper("yi", "Yiddish"),
    TranslationLang("yor_Latn", "Yoruba").Whisper("yo", "Yoruba").SeamlessT_Tx("yor", "Yoruba"),
    TranslationLang("yue_Hant", "Yue Chinese").Whisper("yue", "cantonese").M2M100("zh", "Chinese (zh-yue)").SeamlessT_Tx("yue", "Cantonese"),
    TranslationLang("zho_Hans", "Chinese (Simplified)").Whisper("zh", "Chinese (Simplified)", "Chinese", "mandarin").SeamlessT_Tx("cmn", "Mandarin Chinese (Simplified)"),
    TranslationLang("zho_Hant", "Chinese (Traditional)").Whisper("zh", "Chinese (Traditional)").SeamlessT_Tx("cmn_Hant", "Mandarin Chinese (Traditional)"),
    TranslationLang("zsm_Latn", "Standard Malay").Whisper("ms", "Malay").SeamlessT_Tx("zsm", "Standard Malay").SeamlessS_Sp("zlm", "Colloquial Malay"), #msa:Malay (macrolanguage), zsm:Standard Malay, zlm:Malay (individual language), 
    TranslationLang("zul_Latn", "Zulu").M2M100("zu", "Zulu").SeamlessT_Tx("zul", "Zulu"),
    # TranslationLang(None, None).Whisper("br", "Breton"), # Both whisper and m2m100 support the Breton language, but nllb does not have this language.
]


_TO_LANG_NAME_NLLB = {name.lower(): language for language in TranslationLangs if language.nllb is not None for name in language.nllb.names}

_TO_LANG_NAME_M2M100 = {name.lower(): language for language in TranslationLangs if language.m2m100 is not None for name in language.m2m100.names}

_TO_LANG_NAME_WHISPER = {name.lower(): language for language in TranslationLangs if language.whisper is not None for name in language.whisper.names}

_TO_LANG_NAME_SeamlessTx = {name.lower(): language for language in TranslationLangs if language.seamlessT_Tx is not None for name in language.seamlessT_Tx.names}

_TO_LANG_CODE_WHISPER = {language.whisper.code.lower(): language for language in TranslationLangs if language.whisper is not None and len(language.whisper.code) > 0}


def get_lang_from_nllb_name(nllbName, default=None) -> TranslationLang:
    """Return the TranslationLang from the lang_name_nllb."""
    return _TO_LANG_NAME_NLLB.get(nllbName.lower() if nllbName else None, default)

def get_lang_from_m2m100_name(m2m100Name, default=None) -> TranslationLang:
    """Return the TranslationLang from the lang_name_m2m100 name."""
    return _TO_LANG_NAME_M2M100.get(m2m100Name.lower() if m2m100Name else None, default)

def get_lang_from_whisper_name(whisperName, default=None) -> TranslationLang:
    """Return the TranslationLang from the lang_name_whisper name."""
    return _TO_LANG_NAME_WHISPER.get(whisperName.lower() if whisperName else None, default)

def get_lang_from_seamlessT_Tx_name(seamlessT_TxName, default=None) -> TranslationLang:
    """Return the TranslationLang from the lang_name_seamlessT_Tx name."""
    return _TO_LANG_NAME_SeamlessTx.get(seamlessT_TxName.lower() if seamlessT_TxName else None, default)

def get_lang_from_whisper_code(whisperCode, default=None) -> TranslationLang:
    """Return the TranslationLang from the lang_code_whisper."""
    return _TO_LANG_CODE_WHISPER.get(whisperCode, default)

def get_lang_nllb_names():
    """Return a list of nllb language names."""
    return list(_TO_LANG_NAME_NLLB.keys())

def get_lang_m2m100_names(codes = []):
    """Return a list of m2m100 language names."""
    return list({name.lower(): None for language in TranslationLangs if language.m2m100 is not None and (len(codes) == 0 or any(code in language.m2m100.code for code in codes)) for name in language.m2m100.names}.keys())

def get_lang_seamlessT_Tx_names(codes = []):
    """Return a list of seamlessT_Tx language names."""
    return list({name.lower(): None for language in TranslationLangs if language.seamlessT_Tx is not None and (len(codes) == 0 or any(code in language.seamlessT_Tx.code for code in codes)) for name in language.seamlessT_Tx.names}.keys())

def get_lang_whisper_names():
    """Return a list of whisper language names."""
    return list(_TO_LANG_NAME_WHISPER.keys())

def sort_lang_by_whisper_codes(specified_order: list = []):
    def sort_by_whisper_code(lang: TranslationLang, specified_order: list):
        return (specified_order.index(lang.whisper.code), lang.whisper.names[0]) if lang.whisper.code in specified_order else (len(specified_order), lang.whisper.names[0])

    def cmp_by_whisper_code(lang1: TranslationLang, lang2: TranslationLang):
        val1 = sort_by_whisper_code(lang1, specified_order)
        val2 = sort_by_whisper_code(lang2, specified_order)
        if val1 > val2:
            return 1
        elif val1 == val2:
            return 0
        else: return -1
        
    sorted_translations = sorted(_TO_LANG_NAME_WHISPER.values(), key=cmp_to_key(cmp_by_whisper_code))
    return list({name.lower(): None for language in sorted_translations for name in language.whisper.names}.keys())

if __name__ == "__main__":
    # Test lookup
    print("name:Chinese (Traditional)", get_lang_from_nllb_name("Chinese (Traditional)"))
    print("name:moldavian", get_lang_from_m2m100_name("moldavian"))
    print("code:ja", get_lang_from_whisper_code("ja"))
    print("name:English", get_lang_from_nllb_name('English'))
    print("\n\n")
    
    print(get_lang_m2m100_names(["en", "ja", "zh"]))
    print("\n\n")
    print(sort_lang_by_whisper_codes(["en", "de", "cs", "is", "ru", "zh", "ja"]))