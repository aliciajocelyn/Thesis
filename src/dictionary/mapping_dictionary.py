university_mapping = {
    r'\b(bin(us)?|sunib)\b.*': 'Universitas Bina Nusantara',
    'kemanggisan': 'Universitas Bina Nusantara',
    'alam sutera': 'Universitas Bina Nusantara',
    'malang': 'Universitas Bina Nusantara',

    # BINUS
    r'\b(binus|sunib|bina\s*nusantara)\b.*': 'Universitas Bina Nusantara',

    # UPH
    r'\b(uph|pelita\s*harapan)\b.*': 'Universitas Pelita Harapan',

    # UNTAR
    r'\b(universitas|untar|tarumanagara(\s*university)?)\b.*': 'Universitas Tarumanagara',

    # ITB
    r'\b(itb|institut\s+teknologi\s+bandung)\b.*': 'Institut Teknologi Bandung',

    # UNPAD
    r'\b(unpad|padjadjaran|universitas\s+padjadjaran)\b.*': 'Universitas Padjadjaran',

    # Prasetiya Mulya
    r'\b(prasmul|prasetiya\s+mulya)\b.*': 'Universitas Prasetiya Mulya',

    # UMN
    r'\b(umn|multimedia\s+nusantara)\b.*': 'Universitas Multimedia Nusantara',

    # Atma Jaya
    r'\b(atma\s+jaya|unika\s+atma\s+jaya\s+jkt|atmajaya)\b.*': 'Universitas Atma Jaya',

    # UI
    r'\b(ui|universitas\s+indonesia|indonesia)\b.*': 'Universitas Indonesia',

    # Harvest
    r'\b(stt+i?\s+harvest|harvest\s+international\s+(theology|theological|teological)\s+seminary)\b.*':
        'Harvest International Theological Seminary',

    # UGM
    r'\b(ugm|gadjah\s+mada)\b.*': 'Universitas Gadjah Mada',

    # Kwik Kian Gie
    r'\b(kwik\s+kian\s+gie)\b.*': 'Institut Bisnis dan Informatika Kwik Kian Gie',

    # UBM
    r'\b(ubm|bunda\s+mulia(\s+serpong)?)\b.*': 'Universitas Bunda Mulia',

    # Udayana
    r'\b(udayana)\b.*': 'Universitas Udayana',

    # SGU
    r'\b(sgu|swiss\s+german)\b.*': 'Swiss German University',

    # Sampoerna
    r'\b(sampoerna)\b.*': 'Sampoerna University',

    # IPB
    r'\b(ipb|institut\s+pertanian\s+bogor)\b.*': 'Institut Pertanian Bogor',

    # Ukrida
    r'\b(ukrida)\b.*': 'Universitas Kristen Krida Wacana',

    # ITS
    r'\b(its|institut\s+teknologi\s+sepuluh\s+nopember)\b.*': 'Institut Teknologi Sepuluh Nopember',

    # Petra
    r'\b(petra|kristen\s+petra)\b.*': 'Universitas Kristen Petra',

    # Trisakti
    r'\b(trisakti)\b.*': 'Universitas Trisakti',

    # Dian Nusantara
    r'\b(dian\s+nusantara)\b.*': 'Universitas Dian Nusantara',

    # Sanata Dharma
    r'\b(sanata\s+dharma)\b.*': 'Universitas Sanata Dharma',

    # UNPAR
    r'\b(unpar|parahyangan)\b.*': 'Universitas Katolik Parahyangan',

    # Mercu Buana
    r'\b(mercu\s+buana(\s+yogyakarta)?)\b.*': 'Universitas Mercu Buana',

    # Poltekkes
    r'\b(poltekkes\s+kemenkes\s+bandung)\b.*': 'Politeknik Kesehatan Kemenkes Bandung',

    # UPN Veteran Jakarta
    r'\b(pembangunan\s+nasional\s+veteran\s+jakarta)\b.*': 'Universitas Pembangunan Nasional Veteran Jakarta',

    # UNSIA
    r'\b(unsia|siber\s+asia)\b.*': 'Universitas Siber Asia',

    # PGRI Madiun
    r'\b(pgri\s+madiun)\b.*': 'Universitas PGRI Madiun',

    # Universitas Islam Indonesia
    r'\buii\b': 'Universitas Islam Indonesia',

    # Esa Unggul
    r'\b(esa\s+unggul)\b.*': 'Universitas Esa Unggul',

    # Universitas Diponegoro
    r'\b(undip|diponegoro|universitas\s+diponegoro)\b.*': 'Universitas Diponegoro',

    # INSTIKI
    r'\b(instiki)\b.*': "INSTIKI",

    # Uniji
    r'\b(jakarta\s+international)\b.*': 'University of Jakarta International',

    # UNNC
    r'\bnottingham ningbo china\b': 'University of Nottingham Ningbo China', 

    # Royal Melbourne of Tech
    r'\broyal melbourne institute of technology\b': 'Royal Melbourne Institute of Technology',

    # ASM Ariyanti
    r'\basm ariyanti\b': 'ASM Ariyanti',

    # Universitas Sebelas Maret
    r'\b(uns)\b': 'Universitas Sebelas Maret',

    # ITENAS
    r'\bitenas\b': 'Institut Teknologi Nasional Bandung',

    # Kalbis University
    r'\bkalbis university\b': 'Universitas Kalbis'
}

city_mapping = {
    r'\bjakarta(\s+barat|\s+selatan|\s+timur|\s+pusat)?\b.*': 'Jakarta',
    r'\b(jogj(a|ya|akarta|yakarta)|yogya(karta)?|yogyakarta)\b.*': 'Yogyakarta',
    r'\btangerang\b.*': 'Tangerang',
    r'\bmalang\b.*': 'Malang',
    r'\bningbo\b.*': 'Ningbo',
    r'\bjatinangor\b.*': 'Jatinangor',
    r'\bbandung\b.*': 'Bandung',
    r'\bdenpasar\b.*': 'Denpasar',
    r'\bbekasi\b.*': 'Bekasi',
    r'\bbatam\b.*': 'Batam',
    r'\bsumedang\b.*': 'Sumedang',
    r'\bdepok\b.*': 'Depok',
    r'\bsurabaya\b.*': 'Surabaya',
    r'\bbogor\b.*': 'Bogor',
    r'\bsemarang\b.*': 'Semarang',
    r'\bmadiun\b.*': 'Madiun',
    r'\bbali\b': 'Bali'
}

program_mapping = { 
    r'\b(cs|co(m|n)(ou|pu|ps)?(ter)?\s*s(c|x)i(ence)?|csie|it)\b.*': 'Computer Science',
    r'\b(co(mp|pm)sci)\b': 'Computer Science',
    r'\bIlmu Komputer\b': 'Computer Science',
    r'\b(data\s*s(c|x)i(en|em)?ce)\b.*': 'Data Science',
    r'\bdkv\b.*': 'Desain Komunikasi Visual',
    r'\bs1\s*branding\b.*': 'Branding',
    r'\b(sistem|system|systems)\s*(informasi|information)|si\b.*': 'Sistem Informasi',
    r'\b(informasi|information)\s*(sistem|system|systems)\b.*': 'Sistem Informasi',
    r'\b(teknik\s*informatika|informatika|pjj\s*informatika|ti|pjj\s*teknik\s*informatika)\b.*': 'Computer Science',
    r'\b(arsitektur|ar(s|z)?itek|architecture)\b.*': 'Arsitektur',
    r'\b(teknik\s*industri)\b.*': 'Teknik Industri',
    r'\b(teknik\s*elektro)\b.*': 'Teknik Elektro',
    r'\b(teknologi\s*pangan|food\s*tech(nology)?|tek\s*pangan|teknik\s*pangan)|tekpang\b.*': 'Teknologi Pangan',
    r'\b(gizi(\s*dan\s*dietetika)?|nutrition)\b.*': 'Gizi',
    r'\b(farmasi|sains\s+dan\s+teknologi\s+farmasi)\b.*': 'Farmasi',
    r'\b(kedokteran\s*gigi)\b.*': 'Kedokteran Gigi',
    r'\b(kedokteran(\s*umum)?|fk)\b.*': 'Kedokteran',
    r'\b(psikologi)\b.*': 'Psikologi',
    r'\b(ilmu\s*komunikasi|fakultas\s*ilmu\s*komunikasi|marketing\s*communication|ilkom)\b.*': 'Ilmu Komunikasi',
    r'\b(m(a|e)n(a|e)jem(e|w)n|managemenr|management|manegement|business\s*(management|creation)|global\s*business\s*marketing|business)\b.*': 'Manajemen',
    r'\b(accounting|akuntansi|s1\s*akuntansi)\b.*': 'Akuntansi',
    r'\b(finance)\b.*': 'Keuangan',
    r'\b(hukum)\b.*': 'Hukum',
    r'\b(teologi)\b.*': 'Teologi',
    r'\b(musik\s*gerejawi)\b.*': 'Musik Gerejawi',
    r'\bMusik\b': 'Musik',
    r'\b(bahasa\s*mandarin|bisnis\s*mandarin)\b.*': 'Bahasa dan Bisnis Mandarin',
    r'\b(international\s*culinary\s*business)\b.*': 'International Culinary Business',
    r'\b(international\s*management)\b.*': 'International Management',
    r'\b(perhotelan)\b.*': 'Perhotelan',
    r'\b(biologi)\b.*': 'Biologi',
    r'\b(matematika)\b.*': 'Matematika',
    r'\b(rekayasa\s*pertanian)\b.*': 'Rekayasa Pertanian',
    r'\b(budidaya\s*perairan)\b.*': 'Budidaya Perairan',
    r'\b(mobile\s*(application|app)?(\s*and\s*technology|n\s*tech)?|mobile\s*app\s*n\s*tech)\b.*': 'Mobile Application and Technology',
    r'\b(interactive\s*design\s*and\s*technology)\b.*': 'Interactive Design and Technology',
    r'\b(game\s*application(\s*and\s*technology)?|game\s*application\s*&\s*technology)\b.*': 'Game Application and Technology',
    r'\b(sistem\s*informasi\s*dan\s*akuntansi|information\s*systems\s*&\s*accounting)\b.*': 'Sistem Informasi dan Akuntansi',
    r'\b(the\s*artificial\s*intelligence.*|ai.*cognitive\s*computing)\b.*': 'Artificial Intelligence and Cognitive Computing',
    r'\b(new\s*medi(a)?|film)\b.*': 'New Media & Film',
    r'\b(ekonomi)\b.*': 'Ilmu Ekonomi',
    r'\bteknik\b': "Teknik",
    r'\bsosiologi\b': "Sosiologi",
    r'\b Master Track of Information Technology\b': 'Master Track of Information Technology',
}