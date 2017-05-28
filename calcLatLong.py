from geopy.geocoders import GoogleV3

import csv 
with open("jan17.csv") as f:
	rd = csv.reader(f)
	rows = [r for r in rd]

	print (rows[5][5]+" "+rows[5][6])

	geolocator = GoogleV3()

	cnt = 0
	cnt2 = 0
	for i in range (1,10):
		print(cnt2)
		cnt2 += 1
		location = geolocator.geocode(rows[i][5] + " " + rows[i][6] + " ST" + " Houston", timeout=10)
		print((location.latitude, location.longitude))
		print(location)
		if (location == 'None'):
			cnt = cnt + 1

	print(cnt)