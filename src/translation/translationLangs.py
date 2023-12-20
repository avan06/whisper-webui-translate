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
        self.seamlessTx = None
        
    def Whisper(self, code: str, *names: str):
        self.whisper = Lang(code, *names)
        if self.m2m100 is None:
            self.m2m100 = self.whisper
        return self
    
    def M2M100(self, code: str, name: str):
        self.m2m100 = Lang(code, name)
        return self
    
    def SeamlessTx(self, code: str, name: str):
        self.seamlessTx = Lang(code, name)
        return self

    def __repr__(self):
        result = ""
        if self.nllb:
            result += f"NLLB={self.nllb} "
        if self.whisper:
            result += f"WHISPER={self.whisper} "
        if self.m2m100:
            result += f"M2M100={self.m2m100} "
        if self.seamlessTx:
            result += f"SeamlessTx={self.seamlessTx} "
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
    TranslationLang("ace_Arab", "Acehnese (Arabic script)"),
    TranslationLang("ace_Latn", "Acehnese (Latin script)"),
    TranslationLang("acm_Arab", "Mesopotamian Arabic").Whisper("ar", "Arabic"),
    TranslationLang("acq_Arab", "Ta’izzi-Adeni Arabic").Whisper("ar", "Arabic"),
    TranslationLang("aeb_Arab", "Tunisian Arabic"),
    TranslationLang("afr_Latn", "Afrikaans").Whisper("af", "Afrikaans").SeamlessTx("afr", "Afrikaans"),
    TranslationLang("ajp_Arab", "South Levantine Arabic").Whisper("ar", "Arabic"),
    TranslationLang("aka_Latn", "Akan"),
    TranslationLang("amh_Ethi", "Amharic").Whisper("am", "Amharic").SeamlessTx("amh", "Amharic"),
    TranslationLang("apc_Arab", "North Levantine Arabic").Whisper("ar", "Arabic"),
    TranslationLang("arb_Arab", "Modern Standard Arabic").Whisper("ar", "Arabic").SeamlessTx("arb", "Modern Standard Arabic"),
    TranslationLang("arb_Latn", "Modern Standard Arabic (Romanized)"),
    TranslationLang("ars_Arab", "Najdi Arabic").Whisper("ar", "Arabic"),
    TranslationLang("ary_Arab", "Moroccan Arabic").Whisper("ar", "Arabic").SeamlessTx("ary", "Moroccan Arabic"),
    TranslationLang("arz_Arab", "Egyptian Arabic").Whisper("ar", "Arabic").SeamlessTx("arz", "Egyptian Arabic"),
    TranslationLang("asm_Beng", "Assamese").Whisper("as", "Assamese").SeamlessTx("asm", "Assamese"),
    TranslationLang("ast_Latn", "Asturian").M2M100("ast", "Asturian"),
    TranslationLang("awa_Deva", "Awadhi"),
    TranslationLang("ayr_Latn", "Central Aymara"),
    TranslationLang("azb_Arab", "South Azerbaijani").Whisper("az", "Azerbaijani"),
    TranslationLang("azj_Latn", "North Azerbaijani").Whisper("az", "Azerbaijani").SeamlessTx("azj", "North Azerbaijani"),
    TranslationLang("bak_Cyrl", "Bashkir").Whisper("ba", "Bashkir"),
    TranslationLang("bam_Latn", "Bambara"),
    TranslationLang("ban_Latn", "Balinese"),
    TranslationLang("bel_Cyrl", "Belarusian").Whisper("be", "Belarusian").SeamlessTx("bel", "Belarusian"),
    TranslationLang("bem_Latn", "Bemba"),
    TranslationLang("ben_Beng", "Bengali").Whisper("bn", "Bengali").SeamlessTx("ben", "Bengali"),
    TranslationLang("bho_Deva", "Bhojpuri"),
    TranslationLang("bjn_Arab", "Banjar (Arabic script)"),
    TranslationLang("bjn_Latn", "Banjar (Latin script)"),
    TranslationLang("bod_Tibt", "Standard Tibetan").Whisper("bo", "Tibetan"),
    TranslationLang("bos_Latn", "Bosnian").Whisper("bs", "Bosnian").SeamlessTx("bos", "Bosnian"),
    TranslationLang("bug_Latn", "Buginese"),
    TranslationLang("bul_Cyrl", "Bulgarian").Whisper("bg", "Bulgarian").SeamlessTx("bul", "Bulgarian"),
    TranslationLang("cat_Latn", "Catalan").Whisper("ca", "Catalan", "valencian").SeamlessTx("cat", "Catalan"),
    TranslationLang("ceb_Latn", "Cebuano").M2M100("ceb", "Cebuano").SeamlessTx("ceb", "Cebuano"),
    TranslationLang("ces_Latn", "Czech").Whisper("cs", "Czech").SeamlessTx("ces", "Czech"),
    TranslationLang("cjk_Latn", "Chokwe"),
    TranslationLang("ckb_Arab", "Central Kurdish").SeamlessTx("ckb", "Central Kurdish"),
    TranslationLang("crh_Latn", "Crimean Tatar"),
    TranslationLang("cym_Latn", "Welsh").Whisper("cy", "Welsh").SeamlessTx("cym", "Welsh"),
    TranslationLang("dan_Latn", "Danish").Whisper("da", "Danish").SeamlessTx("dan", "Danish"),
    TranslationLang("deu_Latn", "German").Whisper("de", "German").SeamlessTx("deu", "German"),
    TranslationLang("dik_Latn", "Southwestern Dinka"),
    TranslationLang("dyu_Latn", "Dyula"),
    TranslationLang("dzo_Tibt", "Dzongkha"),
    TranslationLang("ell_Grek", "Greek").Whisper("el", "Greek").SeamlessTx("ell", "Greek"),
    TranslationLang("eng_Latn", "English").Whisper("en", "English").SeamlessTx("eng", "English"),
    TranslationLang("epo_Latn", "Esperanto"),
    TranslationLang("est_Latn", "Estonian").Whisper("et", "Estonian").SeamlessTx("est", "Estonian"),
    TranslationLang("eus_Latn", "Basque").Whisper("eu", "Basque").SeamlessTx("eus", "Basque"),
    TranslationLang("ewe_Latn", "Ewe"),
    TranslationLang("fao_Latn", "Faroese").Whisper("fo", "Faroese"),
    TranslationLang("fij_Latn", "Fijian"),
    TranslationLang("fin_Latn", "Finnish").Whisper("fi", "Finnish").SeamlessTx("fin", "Finnish"),
    TranslationLang("fon_Latn", "Fon"),
    TranslationLang("fra_Latn", "French").Whisper("fr", "French").SeamlessTx("fra", "French"),
    TranslationLang("fur_Latn", "Friulian"),
    TranslationLang("fuv_Latn", "Nigerian Fulfulde").M2M100("ff", "Fulah").SeamlessTx("fuv", "Nigerian Fulfulde"),
    TranslationLang("gla_Latn", "Scottish Gaelic").M2M100("gd", "Scottish Gaelic"),
    TranslationLang("gle_Latn", "Irish").M2M100("ga", "Irish").SeamlessTx("gle", "Irish"),
    TranslationLang("glg_Latn", "Galician").Whisper("gl", "Galician").SeamlessTx("glg", "Galician"),
    TranslationLang("grn_Latn", "Guarani"),
    TranslationLang("guj_Gujr", "Gujarati").Whisper("gu", "Gujarati").SeamlessTx("guj", "Gujarati"),
    TranslationLang("hat_Latn", "Haitian Creole").Whisper("ht", "Haitian creole", "haitian"),
    TranslationLang("hau_Latn", "Hausa").Whisper("ha", "Hausa"),
    TranslationLang("heb_Hebr", "Hebrew").Whisper("he", "Hebrew").SeamlessTx("heb", "Hebrew"),
    TranslationLang("hin_Deva", "Hindi").Whisper("hi", "Hindi").SeamlessTx("hin", "Hindi"),
    TranslationLang("hne_Deva", "Chhattisgarhi"),
    TranslationLang("hrv_Latn", "Croatian").Whisper("hr", "Croatian").SeamlessTx("hrv", "Croatian"),
    TranslationLang("hun_Latn", "Hungarian").Whisper("hu", "Hungarian").SeamlessTx("hun", "Hungarian"),
    TranslationLang("hye_Armn", "Armenian").Whisper("hy", "Armenian").SeamlessTx("hye", "Armenian"),
    TranslationLang("ibo_Latn", "Igbo").M2M100("ig", "Igbo").SeamlessTx("ibo", "Igbo"),
    TranslationLang("ilo_Latn", "Ilocano").M2M100("ilo", "Iloko"),
    TranslationLang("ind_Latn", "Indonesian").Whisper("id", "Indonesian").SeamlessTx("ind", "Indonesian"),
    TranslationLang("isl_Latn", "Icelandic").Whisper("is", "Icelandic").SeamlessTx("isl", "Icelandic"),
    TranslationLang("ita_Latn", "Italian").Whisper("it", "Italian").SeamlessTx("ita", "Italian"),
    TranslationLang("jav_Latn", "Javanese").Whisper("jw", "Javanese").M2M100("jv", "Javanese").SeamlessTx("jav", "Javanese"),
    TranslationLang("jpn_Jpan", "Japanese").Whisper("ja", "Japanese").SeamlessTx("jpn", "Japanese"),
    TranslationLang("kab_Latn", "Kabyle"),
    TranslationLang("kac_Latn", "Jingpho"),
    TranslationLang("kam_Latn", "Kamba"),
    TranslationLang("kan_Knda", "Kannada").Whisper("kn", "Kannada").SeamlessTx("kan", "Kannada"),
    TranslationLang("kas_Arab", "Kashmiri (Arabic script)"),
    TranslationLang("kas_Deva", "Kashmiri (Devanagari script)"),
    TranslationLang("kat_Geor", "Georgian").Whisper("ka", "Georgian").SeamlessTx("kat", "Georgian"),
    TranslationLang("knc_Arab", "Central Kanuri (Arabic script)"),
    TranslationLang("knc_Latn", "Central Kanuri (Latin script)"),
    TranslationLang("kaz_Cyrl", "Kazakh").Whisper("kk", "Kazakh").SeamlessTx("kaz", "Kazakh"),
    TranslationLang("kbp_Latn", "Kabiyè"),
    TranslationLang("kea_Latn", "Kabuverdianu"),
    TranslationLang("khm_Khmr", "Khmer").Whisper("km", "Khmer").SeamlessTx("khm", "Khmer"),
    TranslationLang("kik_Latn", "Kikuyu"),
    TranslationLang("kin_Latn", "Kinyarwanda"),
    TranslationLang("kir_Cyrl", "Kyrgyz").SeamlessTx("kir", "Kyrgyz"),
    TranslationLang("kmb_Latn", "Kimbundu"),
    TranslationLang("kmr_Latn", "Northern Kurdish"),
    TranslationLang("kon_Latn", "Kikongo"),
    TranslationLang("kor_Hang", "Korean").Whisper("ko", "Korean").SeamlessTx("kor", "Korean"),
    TranslationLang("lao_Laoo", "Lao").Whisper("lo", "Lao").SeamlessTx("lao", "Lao"),
    TranslationLang("lij_Latn", "Ligurian"),
    TranslationLang("lim_Latn", "Limburgish"),
    TranslationLang("lin_Latn", "Lingala").Whisper("ln", "Lingala"),
    TranslationLang("lit_Latn", "Lithuanian").Whisper("lt", "Lithuanian").SeamlessTx("lit", "Lithuanian"),
    TranslationLang("lmo_Latn", "Lombard"),
    TranslationLang("ltg_Latn", "Latgalian"),
    TranslationLang("ltz_Latn", "Luxembourgish").Whisper("lb", "Luxembourgish", "letzeburgesch"),
    TranslationLang("lua_Latn", "Luba-Kasai"),
    TranslationLang("lug_Latn", "Ganda").M2M100("lg", "Ganda").SeamlessTx("lug", "Ganda"),
    TranslationLang("luo_Latn", "Luo").SeamlessTx("luo", "Luo"),
    TranslationLang("lus_Latn", "Mizo"),
    TranslationLang("lvs_Latn", "Standard Latvian").Whisper("lv", "Latvian").SeamlessTx("lvs", "Standard Latvian"),
    TranslationLang("mag_Deva", "Magahi"),
    TranslationLang("mai_Deva", "Maithili").SeamlessTx("mai", "Maithili"),
    TranslationLang("mal_Mlym", "Malayalam").Whisper("ml", "Malayalam").SeamlessTx("mal", "Malayalam"),
    TranslationLang("mar_Deva", "Marathi").Whisper("mr", "Marathi").SeamlessTx("mar", "Marathi"),
    TranslationLang("min_Arab", "Minangkabau (Arabic script)"),
    TranslationLang("min_Latn", "Minangkabau (Latin script)"),
    TranslationLang("mkd_Cyrl", "Macedonian").Whisper("mk", "Macedonian").SeamlessTx("mkd", "Macedonian"),
    TranslationLang("plt_Latn", "Plateau Malagasy").Whisper("mg", "Malagasy"),
    TranslationLang("mlt_Latn", "Maltese").Whisper("mt", "Maltese").SeamlessTx("mlt", "Maltese"),
    TranslationLang("mni_Beng", "Meitei (Bengali script)").SeamlessTx("mni", "Meitei"),
    TranslationLang("khk_Cyrl", "Halh Mongolian").Whisper("mn", "Mongolian").SeamlessTx("khk", "Halh Mongolian"),
    TranslationLang("mos_Latn", "Mossi"),
    TranslationLang("mri_Latn", "Maori").Whisper("mi", "Maori"),
    TranslationLang("mya_Mymr", "Burmese").Whisper("my", "Myanmar", "burmese").SeamlessTx("mya", "Burmese"),
    TranslationLang("nld_Latn", "Dutch").Whisper("nl", "Dutch", "flemish").SeamlessTx("nld", "Dutch"),
    TranslationLang("nno_Latn", "Norwegian Nynorsk").Whisper("nn", "Nynorsk").SeamlessTx("nno", "Norwegian Nynorsk"),
    TranslationLang("nob_Latn", "Norwegian Bokmål").Whisper("no", "Norwegian").SeamlessTx("nob", "Norwegian Bokmål"),
    TranslationLang("npi_Deva", "Nepali").Whisper("ne", "Nepali").SeamlessTx("npi", "Nepali"),
    TranslationLang("nso_Latn", "Northern Sotho").M2M100("ns", "Northern Sotho"),
    TranslationLang("nus_Latn", "Nuer"),
    TranslationLang("nya_Latn", "Nyanja").SeamlessTx("nya", "Nyanja"),
    TranslationLang("oci_Latn", "Occitan").Whisper("oc", "Occitan"),
    TranslationLang("gaz_Latn", "West Central Oromo").SeamlessTx("gaz", "West Central Oromo"),
    TranslationLang("ory_Orya", "Odia").M2M100("or", "Oriya").SeamlessTx("ory", "Odia"),
    TranslationLang("pag_Latn", "Pangasinan"),
    TranslationLang("pan_Guru", "Eastern Panjabi").Whisper("pa", "Punjabi", "panjabi").SeamlessTx("pan", "Punjabi"),
    TranslationLang("pap_Latn", "Papiamento"),
    TranslationLang("pes_Arab", "Western Persian").Whisper("fa", "Persian").SeamlessTx("pes", "Western Persian"),
    TranslationLang("pol_Latn", "Polish").Whisper("pl", "Polish").SeamlessTx("pol", "Polish"),
    TranslationLang("por_Latn", "Portuguese").Whisper("pt", "Portuguese").SeamlessTx("por", "Portuguese"),
    TranslationLang("prs_Arab", "Dari"),
    TranslationLang("pbt_Arab", "Southern Pashto").Whisper("ps", "Pashto", "pushto").SeamlessTx("pbt", "Southern Pashto"),
    TranslationLang("quy_Latn", "Ayacucho Quechua"),
    TranslationLang("ron_Latn", "Romanian").Whisper("ro", "Romanian", "moldavian", "moldovan").SeamlessTx("ron", "Romanian"),
    TranslationLang("run_Latn", "Rundi"),
    TranslationLang("rus_Cyrl", "Russian").Whisper("ru", "Russian").SeamlessTx("rus", "Russian"),
    TranslationLang("sag_Latn", "Sango"),
    TranslationLang("san_Deva", "Sanskrit").Whisper("sa", "Sanskrit"),
    TranslationLang("sat_Olck", "Santali"),
    TranslationLang("scn_Latn", "Sicilian"),
    TranslationLang("shn_Mymr", "Shan"),
    TranslationLang("sin_Sinh", "Sinhala").Whisper("si", "Sinhala", "sinhalese"),
    TranslationLang("slk_Latn", "Slovak").Whisper("sk", "Slovak").SeamlessTx("slk", "Slovak"),
    TranslationLang("slv_Latn", "Slovenian").Whisper("sl", "Slovenian").SeamlessTx("slv", "Slovenian"),
    TranslationLang("smo_Latn", "Samoan"),
    TranslationLang("sna_Latn", "Shona").Whisper("sn", "Shona").SeamlessTx("sna", "Shona"),
    TranslationLang("snd_Arab", "Sindhi").Whisper("sd", "Sindhi").SeamlessTx("snd", "Sindhi"),
    TranslationLang("som_Latn", "Somali").Whisper("so", "Somali").SeamlessTx("som", "Somali"),
    TranslationLang("sot_Latn", "Southern Sotho"),
    TranslationLang("spa_Latn", "Spanish").Whisper("es", "Spanish", "castilian").SeamlessTx("spa", "Spanish"),
    TranslationLang("als_Latn", "Tosk Albanian").Whisper("sq", "Albanian"),
    TranslationLang("srd_Latn", "Sardinian"),
    TranslationLang("srp_Cyrl", "Serbian").Whisper("sr", "Serbian").SeamlessTx("srp", "Serbian"),
    TranslationLang("ssw_Latn", "Swati").M2M100("ss", "Swati"),
    TranslationLang("sun_Latn", "Sundanese").Whisper("su", "Sundanese"),
    TranslationLang("swe_Latn", "Swedish").Whisper("sv", "Swedish").SeamlessTx("swe", "Swedish"),
    TranslationLang("swh_Latn", "Swahili").Whisper("sw", "Swahili").SeamlessTx("swh", "Swahili"),
    TranslationLang("szl_Latn", "Silesian"),
    TranslationLang("tam_Taml", "Tamil").Whisper("ta", "Tamil").SeamlessTx("tam", "Tamil"),
    TranslationLang("tat_Cyrl", "Tatar").Whisper("tt", "Tatar"),
    TranslationLang("tel_Telu", "Telugu").Whisper("te", "Telugu").SeamlessTx("tel", "Telugu"),
    TranslationLang("tgk_Cyrl", "Tajik").Whisper("tg", "Tajik").SeamlessTx("tgk", "Tajik"),
    TranslationLang("tgl_Latn", "Tagalog").Whisper("tl", "Tagalog").SeamlessTx("tgl", "Tagalog"),
    TranslationLang("tha_Thai", "Thai").Whisper("th", "Thai").SeamlessTx("tha", "Thai"),
    TranslationLang("tir_Ethi", "Tigrinya"),
    TranslationLang("taq_Latn", "Tamasheq (Latin script)"),
    TranslationLang("taq_Tfng", "Tamasheq (Tifinagh script)"),
    TranslationLang("tpi_Latn", "Tok Pisin"),
    TranslationLang("tsn_Latn", "Tswana").M2M100("tn", "Tswana"),
    TranslationLang("tso_Latn", "Tsonga"),
    TranslationLang("tuk_Latn", "Turkmen").Whisper("tk", "Turkmen"),
    TranslationLang("tum_Latn", "Tumbuka"),
    TranslationLang("tur_Latn", "Turkish").Whisper("tr", "Turkish").SeamlessTx("tur", "Turkish"),
    TranslationLang("twi_Latn", "Twi"),
    TranslationLang("tzm_Tfng", "Central Atlas Tamazight"),
    TranslationLang("uig_Arab", "Uyghur"),
    TranslationLang("ukr_Cyrl", "Ukrainian").Whisper("uk", "Ukrainian").SeamlessTx("ukr", "Ukrainian"),
    TranslationLang("umb_Latn", "Umbundu"),
    TranslationLang("urd_Arab", "Urdu").Whisper("ur", "Urdu").SeamlessTx("urd", "Urdu"),
    TranslationLang("uzn_Latn", "Northern Uzbek").Whisper("uz", "Uzbek").SeamlessTx("uzn", "Northern Uzbek"),
    TranslationLang("vec_Latn", "Venetian"),
    TranslationLang("vie_Latn", "Vietnamese").Whisper("vi", "Vietnamese").SeamlessTx("vie", "Vietnamese"),
    TranslationLang("war_Latn", "Waray"),
    TranslationLang("wol_Latn", "Wolof").M2M100("wo", "Wolof"),
    TranslationLang("xho_Latn", "Xhosa").M2M100("xh", "Xhosa"),
    TranslationLang("ydd_Hebr", "Eastern Yiddish").Whisper("yi", "Yiddish"),
    TranslationLang("yor_Latn", "Yoruba").Whisper("yo", "Yoruba").SeamlessTx("yor", "Yoruba"),
    TranslationLang("yue_Hant", "Yue Chinese").Whisper("yue", "cantonese").M2M100("zh", "Chinese (zh-yue)").SeamlessTx("yue", "Cantonese"),
    TranslationLang("zho_Hans", "Chinese (Simplified)").Whisper("zh", "Chinese (Simplified)", "Chinese", "mandarin").SeamlessTx("cmn", "Mandarin Chinese (Simplified)"),
    TranslationLang("zho_Hant", "Chinese (Traditional)").Whisper("zh", "Chinese (Traditional)").SeamlessTx("cmn_Hant", "Mandarin Chinese (Traditional)"),
    TranslationLang("zsm_Latn", "Standard Malay").Whisper("ms", "Malay").SeamlessTx("zsm", "Standard Malay"),
    TranslationLang("zul_Latn", "Zulu").M2M100("zu", "Zulu").SeamlessTx("zul", "Zulu"),
    # TranslationLang(None, None).Whisper("br", "Breton"), # Both whisper and m2m100 support the Breton language, but nllb does not have this language.
]


_TO_LANG_NAME_NLLB = {name.lower(): language for language in TranslationLangs if language.nllb is not None for name in language.nllb.names}

_TO_LANG_NAME_M2M100 = {name.lower(): language for language in TranslationLangs if language.m2m100 is not None for name in language.m2m100.names}

_TO_LANG_NAME_WHISPER = {name.lower(): language for language in TranslationLangs if language.whisper is not None for name in language.whisper.names}

_TO_LANG_NAME_SeamlessTx = {name.lower(): language for language in TranslationLangs if language.seamlessTx is not None for name in language.seamlessTx.names}

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

def get_lang_from_seamlessTx_name(seamlessTxName, default=None) -> TranslationLang:
    """Return the TranslationLang from the lang_name_seamlessTx name."""
    return _TO_LANG_NAME_SeamlessTx.get(seamlessTxName.lower() if seamlessTxName else None, default)

def get_lang_from_whisper_code(whisperCode, default=None) -> TranslationLang:
    """Return the TranslationLang from the lang_code_whisper."""
    return _TO_LANG_CODE_WHISPER.get(whisperCode, default)

def get_lang_nllb_names():
    """Return a list of nllb language names."""
    return list(_TO_LANG_NAME_NLLB.keys())

def get_lang_m2m100_names(codes = []):
    """Return a list of m2m100 language names."""
    return list({name.lower(): None for language in TranslationLangs if language.m2m100 is not None and (len(codes) == 0 or any(code in language.m2m100.code for code in codes)) for name in language.m2m100.names}.keys())

def get_lang_seamlessTx_names(codes = []):
    """Return a list of seamlessTx language names."""
    return list({name.lower(): None for language in TranslationLangs if language.seamlessTx is not None and (len(codes) == 0 or any(code in language.seamlessTx.code for code in codes)) for name in language.seamlessTx.names}.keys())

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