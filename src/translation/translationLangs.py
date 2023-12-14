from functools import cmp_to_key

class Lang():
    def __init__(self, code: str, *names: str):
        self.code = code
        self.names = names
        
    def __repr__(self):
        return f"code:{self.code}, name:{self.names}"

class TranslationLang():
    def __init__(self, nllb: Lang, whisper: Lang = None, m2m100: Lang = None):
        self.nllb    = nllb
        self.whisper = whisper
        self.m2m100  = None
        
        if m2m100 is None: m2m100 = whisper
        if m2m100 is not None and len(m2m100.names) > 0:
            self.m2m100  = m2m100

    def __repr__(self):
        result = ""
        if self.nllb is not None:
            result += f"NLLB={self.nllb} "
        if self.whisper is not None:
            result += f"WHISPER={self.whisper} "
        if self.m2m100 is not None:
            result += f"M@M100={self.m2m100} "
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
"""
TranslationLangs = [
    TranslationLang(Lang("ace_Arab", "Acehnese (Arabic script)")),
    TranslationLang(Lang("ace_Latn", "Acehnese (Latin script)")),
    TranslationLang(Lang("acm_Arab", "Mesopotamian Arabic"), Lang("ar", "Arabic")),
    TranslationLang(Lang("acq_Arab", "Ta’izzi-Adeni Arabic"), Lang("ar", "Arabic")),
    TranslationLang(Lang("aeb_Arab", "Tunisian Arabic")),
    TranslationLang(Lang("afr_Latn", "Afrikaans"), Lang("af", "Afrikaans")),
    TranslationLang(Lang("ajp_Arab", "South Levantine Arabic"), Lang("ar", "Arabic")),
    TranslationLang(Lang("aka_Latn", "Akan")),
    TranslationLang(Lang("amh_Ethi", "Amharic"), Lang("am", "Amharic")),
    TranslationLang(Lang("apc_Arab", "North Levantine Arabic"), Lang("ar", "Arabic")),
    TranslationLang(Lang("arb_Arab", "Modern Standard Arabic"), Lang("ar", "Arabic")),
    TranslationLang(Lang("arb_Latn", "Modern Standard Arabic (Romanized)")),
    TranslationLang(Lang("ars_Arab", "Najdi Arabic"), Lang("ar", "Arabic")),
    TranslationLang(Lang("ary_Arab", "Moroccan Arabic"), Lang("ar", "Arabic")),
    TranslationLang(Lang("arz_Arab", "Egyptian Arabic"), Lang("ar", "Arabic")),
    TranslationLang(Lang("asm_Beng", "Assamese"), Lang("as", "Assamese")),
    TranslationLang(Lang("ast_Latn", "Asturian"), None, Lang("ast", "Asturian")),
    TranslationLang(Lang("awa_Deva", "Awadhi")),
    TranslationLang(Lang("ayr_Latn", "Central Aymara")),
    TranslationLang(Lang("azb_Arab", "South Azerbaijani"), Lang("az", "Azerbaijani")),
    TranslationLang(Lang("azj_Latn", "North Azerbaijani"), Lang("az", "Azerbaijani")),
    TranslationLang(Lang("bak_Cyrl", "Bashkir"), Lang("ba", "Bashkir")),
    TranslationLang(Lang("bam_Latn", "Bambara")),
    TranslationLang(Lang("ban_Latn", "Balinese")),
    TranslationLang(Lang("bel_Cyrl", "Belarusian"), Lang("be", "Belarusian")),
    TranslationLang(Lang("bem_Latn", "Bemba")),
    TranslationLang(Lang("ben_Beng", "Bengali"), Lang("bn", "Bengali")),
    TranslationLang(Lang("bho_Deva", "Bhojpuri")),
    TranslationLang(Lang("bjn_Arab", "Banjar (Arabic script)")),
    TranslationLang(Lang("bjn_Latn", "Banjar (Latin script)")),
    TranslationLang(Lang("bod_Tibt", "Standard Tibetan"), Lang("bo", "Tibetan")),
    TranslationLang(Lang("bos_Latn", "Bosnian"), Lang("bs", "Bosnian")),
    TranslationLang(Lang("bug_Latn", "Buginese")),
    TranslationLang(Lang("bul_Cyrl", "Bulgarian"), Lang("bg", "Bulgarian")),
    TranslationLang(Lang("cat_Latn", "Catalan"), Lang("ca", "Catalan", "valencian")),
    TranslationLang(Lang("ceb_Latn", "Cebuano"), None, Lang("ceb", "Cebuano")),
    TranslationLang(Lang("ces_Latn", "Czech"), Lang("cs", "Czech")),
    TranslationLang(Lang("cjk_Latn", "Chokwe")),
    TranslationLang(Lang("ckb_Arab", "Central Kurdish")),
    TranslationLang(Lang("crh_Latn", "Crimean Tatar")),
    TranslationLang(Lang("cym_Latn", "Welsh"), Lang("cy", "Welsh")),
    TranslationLang(Lang("dan_Latn", "Danish"), Lang("da", "Danish")),
    TranslationLang(Lang("deu_Latn", "German"), Lang("de", "German")),
    TranslationLang(Lang("dik_Latn", "Southwestern Dinka")),
    TranslationLang(Lang("dyu_Latn", "Dyula")),
    TranslationLang(Lang("dzo_Tibt", "Dzongkha")),
    TranslationLang(Lang("ell_Grek", "Greek"), Lang("el", "Greek")),
    TranslationLang(Lang("eng_Latn", "English"), Lang("en", "English")),
    TranslationLang(Lang("epo_Latn", "Esperanto")),
    TranslationLang(Lang("est_Latn", "Estonian"), Lang("et", "Estonian")),
    TranslationLang(Lang("eus_Latn", "Basque"), Lang("eu", "Basque")),
    TranslationLang(Lang("ewe_Latn", "Ewe")),
    TranslationLang(Lang("fao_Latn", "Faroese"), Lang("fo", "Faroese")),
    TranslationLang(Lang("fij_Latn", "Fijian")),
    TranslationLang(Lang("fin_Latn", "Finnish"), Lang("fi", "Finnish")),
    TranslationLang(Lang("fon_Latn", "Fon")),
    TranslationLang(Lang("fra_Latn", "French"), Lang("fr", "French")),
    TranslationLang(Lang("fur_Latn", "Friulian")),
    TranslationLang(Lang("fuv_Latn", "Nigerian Fulfulde"), None, Lang("ff", "Fulah")),
    TranslationLang(Lang("gla_Latn", "Scottish Gaelic"), None, Lang("gd", "Scottish Gaelic")),
    TranslationLang(Lang("gle_Latn", "Irish"), None, Lang("ga", "Irish")),
    TranslationLang(Lang("glg_Latn", "Galician"), Lang("gl", "Galician")),
    TranslationLang(Lang("grn_Latn", "Guarani")),
    TranslationLang(Lang("guj_Gujr", "Gujarati"), Lang("gu", "Gujarati")),
    TranslationLang(Lang("hat_Latn", "Haitian Creole"), Lang("ht", "Haitian creole", "haitian")),
    TranslationLang(Lang("hau_Latn", "Hausa"), Lang("ha", "Hausa")),
    TranslationLang(Lang("heb_Hebr", "Hebrew"), Lang("he", "Hebrew")),
    TranslationLang(Lang("hin_Deva", "Hindi"), Lang("hi", "Hindi")),
    TranslationLang(Lang("hne_Deva", "Chhattisgarhi")),
    TranslationLang(Lang("hrv_Latn", "Croatian"), Lang("hr", "Croatian")),
    TranslationLang(Lang("hun_Latn", "Hungarian"), Lang("hu", "Hungarian")),
    TranslationLang(Lang("hye_Armn", "Armenian"), Lang("hy", "Armenian")),
    TranslationLang(Lang("ibo_Latn", "Igbo"), None, Lang("ig", "Igbo")),
    TranslationLang(Lang("ilo_Latn", "Ilocano"), None, Lang("ilo", "Iloko")),
    TranslationLang(Lang("ind_Latn", "Indonesian"), Lang("id", "Indonesian")),
    TranslationLang(Lang("isl_Latn", "Icelandic"), Lang("is", "Icelandic")),
    TranslationLang(Lang("ita_Latn", "Italian"), Lang("it", "Italian")),
    TranslationLang(Lang("jav_Latn", "Javanese"), Lang("jw", "Javanese"), Lang("jv", "Javanese")),
    TranslationLang(Lang("jpn_Jpan", "Japanese"), Lang("ja", "Japanese")),
    TranslationLang(Lang("kab_Latn", "Kabyle")),
    TranslationLang(Lang("kac_Latn", "Jingpho")),
    TranslationLang(Lang("kam_Latn", "Kamba")),
    TranslationLang(Lang("kan_Knda", "Kannada"), Lang("kn", "Kannada")),
    TranslationLang(Lang("kas_Arab", "Kashmiri (Arabic script)")),
    TranslationLang(Lang("kas_Deva", "Kashmiri (Devanagari script)")),
    TranslationLang(Lang("kat_Geor", "Georgian"), Lang("ka", "Georgian")),
    TranslationLang(Lang("knc_Arab", "Central Kanuri (Arabic script)")),
    TranslationLang(Lang("knc_Latn", "Central Kanuri (Latin script)")),
    TranslationLang(Lang("kaz_Cyrl", "Kazakh"), Lang("kk", "Kazakh")),
    TranslationLang(Lang("kbp_Latn", "Kabiyè")),
    TranslationLang(Lang("kea_Latn", "Kabuverdianu")),
    TranslationLang(Lang("khm_Khmr", "Khmer"), Lang("km", "Khmer")),
    TranslationLang(Lang("kik_Latn", "Kikuyu")),
    TranslationLang(Lang("kin_Latn", "Kinyarwanda")),
    TranslationLang(Lang("kir_Cyrl", "Kyrgyz")),
    TranslationLang(Lang("kmb_Latn", "Kimbundu")),
    TranslationLang(Lang("kmr_Latn", "Northern Kurdish")),
    TranslationLang(Lang("kon_Latn", "Kikongo")),
    TranslationLang(Lang("kor_Hang", "Korean"), Lang("ko", "Korean")),
    TranslationLang(Lang("lao_Laoo", "Lao"), Lang("lo", "Lao")),
    TranslationLang(Lang("lij_Latn", "Ligurian")),
    TranslationLang(Lang("lim_Latn", "Limburgish")),
    TranslationLang(Lang("lin_Latn", "Lingala"), Lang("ln", "Lingala")),
    TranslationLang(Lang("lit_Latn", "Lithuanian"), Lang("lt", "Lithuanian")),
    TranslationLang(Lang("lmo_Latn", "Lombard")),
    TranslationLang(Lang("ltg_Latn", "Latgalian")),
    TranslationLang(Lang("ltz_Latn", "Luxembourgish"), Lang("lb", "Luxembourgish", "letzeburgesch")),
    TranslationLang(Lang("lua_Latn", "Luba-Kasai")),
    TranslationLang(Lang("lug_Latn", "Ganda"), None, Lang("lg", "Ganda")),
    TranslationLang(Lang("luo_Latn", "Luo")),
    TranslationLang(Lang("lus_Latn", "Mizo")),
    TranslationLang(Lang("lvs_Latn", "Standard Latvian"), Lang("lv", "Latvian")),
    TranslationLang(Lang("mag_Deva", "Magahi")),
    TranslationLang(Lang("mai_Deva", "Maithili")),
    TranslationLang(Lang("mal_Mlym", "Malayalam"), Lang("ml", "Malayalam")),
    TranslationLang(Lang("mar_Deva", "Marathi"), Lang("mr", "Marathi")),
    TranslationLang(Lang("min_Arab", "Minangkabau (Arabic script)")),
    TranslationLang(Lang("min_Latn", "Minangkabau (Latin script)")),
    TranslationLang(Lang("mkd_Cyrl", "Macedonian"), Lang("mk", "Macedonian")),
    TranslationLang(Lang("plt_Latn", "Plateau Malagasy"), Lang("mg", "Malagasy")),
    TranslationLang(Lang("mlt_Latn", "Maltese"), Lang("mt", "Maltese")),
    TranslationLang(Lang("mni_Beng", "Meitei (Bengali script)")),
    TranslationLang(Lang("khk_Cyrl", "Halh Mongolian"), Lang("mn", "Mongolian")),
    TranslationLang(Lang("mos_Latn", "Mossi")),
    TranslationLang(Lang("mri_Latn", "Maori"), Lang("mi", "Maori")),
    TranslationLang(Lang("mya_Mymr", "Burmese"), Lang("my", "Myanmar", "burmese")),
    TranslationLang(Lang("nld_Latn", "Dutch"), Lang("nl", "Dutch", "flemish")),
    TranslationLang(Lang("nno_Latn", "Norwegian Nynorsk"), Lang("nn", "Nynorsk")),
    TranslationLang(Lang("nob_Latn", "Norwegian Bokmål"), Lang("no", "Norwegian")),
    TranslationLang(Lang("npi_Deva", "Nepali"), Lang("ne", "Nepali")),
    TranslationLang(Lang("nso_Latn", "Northern Sotho"), None, Lang("ns", "Northern Sotho")),
    TranslationLang(Lang("nus_Latn", "Nuer")),
    TranslationLang(Lang("nya_Latn", "Nyanja")),
    TranslationLang(Lang("oci_Latn", "Occitan"), Lang("oc", "Occitan")),
    TranslationLang(Lang("gaz_Latn", "West Central Oromo")),
    TranslationLang(Lang("ory_Orya", "Odia"), None, Lang("or", "Oriya")),
    TranslationLang(Lang("pag_Latn", "Pangasinan")),
    TranslationLang(Lang("pan_Guru", "Eastern Panjabi"), Lang("pa", "Punjabi", "panjabi")),
    TranslationLang(Lang("pap_Latn", "Papiamento")),
    TranslationLang(Lang("pes_Arab", "Western Persian"), Lang("fa", "Persian")),
    TranslationLang(Lang("pol_Latn", "Polish"), Lang("pl", "Polish")),
    TranslationLang(Lang("por_Latn", "Portuguese"), Lang("pt", "Portuguese")),
    TranslationLang(Lang("prs_Arab", "Dari")),
    TranslationLang(Lang("pbt_Arab", "Southern Pashto"), Lang("ps", "Pashto", "pushto")),
    TranslationLang(Lang("quy_Latn", "Ayacucho Quechua")),
    TranslationLang(Lang("ron_Latn", "Romanian"), Lang("ro", "Romanian", "moldavian", "moldovan")),
    TranslationLang(Lang("run_Latn", "Rundi")),
    TranslationLang(Lang("rus_Cyrl", "Russian"), Lang("ru", "Russian")),
    TranslationLang(Lang("sag_Latn", "Sango")),
    TranslationLang(Lang("san_Deva", "Sanskrit"), Lang("sa", "Sanskrit")),
    TranslationLang(Lang("sat_Olck", "Santali")),
    TranslationLang(Lang("scn_Latn", "Sicilian")),
    TranslationLang(Lang("shn_Mymr", "Shan")),
    TranslationLang(Lang("sin_Sinh", "Sinhala"), Lang("si", "Sinhala", "sinhalese")),
    TranslationLang(Lang("slk_Latn", "Slovak"), Lang("sk", "Slovak")),
    TranslationLang(Lang("slv_Latn", "Slovenian"), Lang("sl", "Slovenian")),
    TranslationLang(Lang("smo_Latn", "Samoan")),
    TranslationLang(Lang("sna_Latn", "Shona"), Lang("sn", "Shona")),
    TranslationLang(Lang("snd_Arab", "Sindhi"), Lang("sd", "Sindhi")),
    TranslationLang(Lang("som_Latn", "Somali"), Lang("so", "Somali")),
    TranslationLang(Lang("sot_Latn", "Southern Sotho")),
    TranslationLang(Lang("spa_Latn", "Spanish"), Lang("es", "Spanish", "castilian")),
    TranslationLang(Lang("als_Latn", "Tosk Albanian"), Lang("sq", "Albanian")),
    TranslationLang(Lang("srd_Latn", "Sardinian")),
    TranslationLang(Lang("srp_Cyrl", "Serbian"), Lang("sr", "Serbian")),
    TranslationLang(Lang("ssw_Latn", "Swati"), None, Lang("ss", "Swati")),
    TranslationLang(Lang("sun_Latn", "Sundanese"), Lang("su", "Sundanese")),
    TranslationLang(Lang("swe_Latn", "Swedish"), Lang("sv", "Swedish")),
    TranslationLang(Lang("swh_Latn", "Swahili"), Lang("sw", "Swahili")),
    TranslationLang(Lang("szl_Latn", "Silesian")),
    TranslationLang(Lang("tam_Taml", "Tamil"), Lang("ta", "Tamil")),
    TranslationLang(Lang("tat_Cyrl", "Tatar"), Lang("tt", "Tatar")),
    TranslationLang(Lang("tel_Telu", "Telugu"), Lang("te", "Telugu")),
    TranslationLang(Lang("tgk_Cyrl", "Tajik"), Lang("tg", "Tajik")),
    TranslationLang(Lang("tgl_Latn", "Tagalog"), Lang("tl", "Tagalog")),
    TranslationLang(Lang("tha_Thai", "Thai"), Lang("th", "Thai")),
    TranslationLang(Lang("tir_Ethi", "Tigrinya")),
    TranslationLang(Lang("taq_Latn", "Tamasheq (Latin script)")),
    TranslationLang(Lang("taq_Tfng", "Tamasheq (Tifinagh script)")),
    TranslationLang(Lang("tpi_Latn", "Tok Pisin")),
    TranslationLang(Lang("tsn_Latn", "Tswana"), None, Lang("tn", "Tswana")),
    TranslationLang(Lang("tso_Latn", "Tsonga")),
    TranslationLang(Lang("tuk_Latn", "Turkmen"), Lang("tk", "Turkmen")),
    TranslationLang(Lang("tum_Latn", "Tumbuka")),
    TranslationLang(Lang("tur_Latn", "Turkish"), Lang("tr", "Turkish")),
    TranslationLang(Lang("twi_Latn", "Twi")),
    TranslationLang(Lang("tzm_Tfng", "Central Atlas Tamazight")),
    TranslationLang(Lang("uig_Arab", "Uyghur")),
    TranslationLang(Lang("ukr_Cyrl", "Ukrainian"), Lang("uk", "Ukrainian")),
    TranslationLang(Lang("umb_Latn", "Umbundu")),
    TranslationLang(Lang("urd_Arab", "Urdu"), Lang("ur", "Urdu")),
    TranslationLang(Lang("uzn_Latn", "Northern Uzbek"), Lang("uz", "Uzbek")),
    TranslationLang(Lang("vec_Latn", "Venetian")),
    TranslationLang(Lang("vie_Latn", "Vietnamese"), Lang("vi", "Vietnamese")),
    TranslationLang(Lang("war_Latn", "Waray")),
    TranslationLang(Lang("wol_Latn", "Wolof"), None, Lang("wo", "Wolof")),
    TranslationLang(Lang("xho_Latn", "Xhosa"), None, Lang("xh", "Xhosa")),
    TranslationLang(Lang("ydd_Hebr", "Eastern Yiddish"), Lang("yi", "Yiddish")),
    TranslationLang(Lang("yor_Latn", "Yoruba"), Lang("yo", "Yoruba")),
    TranslationLang(Lang("yue_Hant", "Yue Chinese"), Lang("yue", "cantonese"), Lang("zh", "Chinese (zh-yue)")),
    TranslationLang(Lang("zho_Hans", "Chinese (Simplified)"), Lang("zh", "Chinese (Simplified)", "Chinese", "mandarin")),
    TranslationLang(Lang("zho_Hant", "Chinese (Traditional)"), Lang("zh", "Chinese (Traditional)")),
    TranslationLang(Lang("zsm_Latn", "Standard Malay"), Lang("ms", "Malay")),
    TranslationLang(Lang("zul_Latn", "Zulu"), None, Lang("zu", "Zulu")),
    TranslationLang(None, Lang("br", "Breton")), # Both whisper and m2m100 support the Breton language, but nllb does not have this language.
]


_TO_LANG_NAME_NLLB = {name.lower(): language for language in TranslationLangs if language.nllb is not None for name in language.nllb.names}

_TO_LANG_NAME_M2M100 = {name.lower(): language for language in TranslationLangs if language.m2m100 is not None for name in language.m2m100.names}

_TO_LANG_NAME_WHISPER = {name.lower(): language for language in TranslationLangs if language.whisper is not None for name in language.whisper.names}

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

def get_lang_from_whisper_code(whisperCode, default=None) -> TranslationLang:
    """Return the TranslationLang from the lang_code_whisper."""
    return _TO_LANG_CODE_WHISPER.get(whisperCode, default)

def get_lang_nllb_names():
    """Return a list of nllb language names."""
    return list(_TO_LANG_NAME_NLLB.keys())

def get_lang_m2m100_names(codes = []):
    """Return a list of m2m100 language names."""
    return list({name.lower(): None for language in TranslationLangs if language.m2m100 is not None and (len(codes) == 0 or any(code in language.m2m100.code for code in codes)) for name in language.m2m100.names}.keys())

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