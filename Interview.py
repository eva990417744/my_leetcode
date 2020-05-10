from igramscraper.instagram import Instagram

proxies = {
    "http": "http://192.187.125.234:19005",
    "https": "http://192.187.125.234:19005",
}
instagram = Instagram()
instagram.set_proxies(proxies)
a = instagram.get_account('dhttong')

