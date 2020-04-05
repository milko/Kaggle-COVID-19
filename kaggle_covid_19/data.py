################################################################################
# Data management scripts.                                                     #
#                                                                              #
# Import data into database and export it to a Pandas DataFrame.               #
################################################################################

# Import libraries.
import re
import pandas as pd
from arango import ArangoClient


def coders_against_covid(
	host='http://localhost:8529',
	base='Kaggle-Covid-19',
	user='root',
	pswd=None,
	file=None,
	find='FOR d IN `coders_against_covid` RETURN d'
):
	'''
	Manage the coders_against_covid crowd-sourced-covid-19-testing-locations.csv
	dataset.

	This function will:
	- Load the dataset into a Pandas DataFrame,
	- Load the dataset ino an ArangoDB database,
	- Query the dataset form the database and return a Pandas DataFrame.

	Parameters
	----------
	file : str, optional
		The path to the dataset CSV file. The data will be converted to a Pandas
		DataFrame and either returned, if `pswd` was not provided, or stored in
		the database if `pswd` was provided. If this parameter is omitted, it
		is assumed the dataset is stored in the database.
	host : str, optional
		The database host, defaults to "http://localhost:8529"
	base : str, optional
		The database name, defaults to "Kaggle-Covid-19"
	user : str, optional
		The database username, defaults to "root"
	pswd : str, optional
		The database user password. This parameter must be provided if the
		database is to be used.
	find : str, optional
		This parameter represents an AQL query to the database. If the `file`
		parameter is not provided, this parameter determines what gets retrieved
		from the database; by default all records.

	Returns
	-------
	DataFrame
		The dataset pandas DataFrame

	Behaviour:

	The function can be used to retrieve the dataset as a Pandas DataFrame or
	store it into an ArangoDB database.

	If the `file` parameter is provided, the data will be retrieved from the
	CSV file, if the parameter is omitted, the data will be retrieved from the
	database.

	If the `pswd` parameter is provided, it indicates that the function should
	connect with the database: if the `file` parameter was also provided, the
	data in the provided file path will be stored in the database, if the `file`
	parameter was not provided, it means that we want to query the database and
	the `find` parameter will be used as the AQL query.

	When storing the dataset into the database, the resulting dataset will have
	an additional column added: `country_iso`, which represents the ISO 3166-3
	code of the country, in this case 'USA; also the `location_id` column
	value will be copied to the `_key` field.

	The dataset will be stored in the `coders_against_covid` collection. The
	list of dataset columns is stored in the `index` collection as follows:

	{
		_key: 'coders_against_covid',
		columns: [<dataset/collection column names>]
	}
	'''

	# Globals
	kCollection_name = 'coders_against_covid'

	###
	# Load from file.
	###
	if file is not None:

		# Load from file
		df = pd.read_csv(file)

		# Add country.
		df['country_iso'] = 'USA'

		# Normalise boolean fields.
		bool_fields = [
			'is_verified', 'is_hidden', 'is_location_screening_patients',
			'is_location_collecting_specimens',
			'is_location_accepting_third_party_orders_for_testing',
			'is_location_only_testing_patients_that_meet_criteria',
			'is_location_by_appointment_only', 'is_ready_for_prod'
		]

		for field in bool_fields:
			df[field] = df[field].apply(
				lambda x:
					True if x == 't'
					else (
						False if x == 'f'
						else None
					)
			)

		# Normalise geometry field.
		df['geometry'] = df['geometry'].apply(
			lambda x:
				None if pd.isna(x)
				else dict(
					type='Point',
					coordinates=[
						re.search('POINT \((.+) (.+)\)', x).group(1),
						re.search('POINT \((.+) (.+)\)', x).group(2)
					]
				)
		)

		# Return DataFrame if password not provided.
		if pswd is None:
			return df													# ==>

		###
		# Store in database
		###
		client = ArangoClient(hosts=host)
		db = client.db(base, username=user, password=pswd)

		if db.has_collection(kCollection_name):
			collection = db.collection(kCollection_name)
		else:
			collection = db.create_collection(kCollection_name, edge=False)
		if db.has_collection('index'):
			index = db.collection('index')
		else:
			index = db.create_collection('index', edge=False)

		# Load dataset columns.
		if index.has(kCollection_name):
			index.update(dict(
				_key=kCollection_name,
				columns=list(df.columns)
			))
		else:
			index.insert(dict(
				_key=kCollection_name,
				columns=list(df.columns)
			))

		# Copy location_id to _key
		df['_key'] = df['location_id']

		# Load dataset.
		collection.truncate()
		collection.insert_many(
			[
				{k:v for k,v in m.items() if pd.notnull(v)}
				for m in df.to_dict(orient='rows')
			],
			sync=True,
			silent=True
		)

		return df.drop('_key', axis=1)									# ==>

	###
	# Retrieve from database
	###
	client = ArangoClient(hosts=host)
	db = client.db(base, username=user, password=pswd)
	cursor = db.aql.execute(find)

	# Return DataFrame.
	return pd.DataFrame.from_records(
		[document for document in cursor],
		exclude=['_key', '_id', '_rev']
	)  																	# ==>

def county_health_rankings(
	host='http://localhost:8529',
	base='Kaggle-Covid-19',
	user='root',
	pswd=None,
	file=None,
	find='FOR d IN `county_health_rankings` RETURN d'
):
	'''
	Manage the county_health_rankings us-county-health-rankings-2020.csv
	dataset.

	This function will:
	- Load the dataset into a Pandas DataFrame,
	- Load the dataset ino an ArangoDB database,
	- Query the dataset form the database and return a Pandas DataFrame.

	Parameters
	----------
	file : str, optional
		The path to the dataset CSV file. The data will be converted to a Pandas
		DataFrame and either returned, if `pswd` was not provided, or stored in
		the database if `pswd` was provided. If this parameter is omitted, it
		is assumed the dataset is stored in the database.
	host : str, optional
		The database host, defaults to "http://localhost:8529"
	base : str, optional
		The database name, defaults to "Kaggle-Covid-19"
	user : str, optional
		The database username, defaults to "root"
	pswd : str, optional
		The database user password. This parameter must be provided if the
		database is to be used.
	find : str, optional
		This parameter represents an AQL query to the database. If the `file`
		parameter is not provided, this parameter determines what gets retrieved
		from the database; by default all records.

	Returns
	-------
	DataFrame
		The dataset pandas DataFrame

	Behaviour:

	The function can be used to retrieve the dataset as a Pandas DataFrame or
	store it into an ArangoDB database.

	If the `file` parameter is provided, the data will be retrieved from the
	CSV file, if the parameter is omitted, the data will be retrieved from the
	database.

	If the `pswd` parameter is provided, it indicates that the function should
	connect with the database: if the `file` parameter was also provided, the
	data in the provided file path will be stored in the database, if the `file`
	parameter was not provided, it means that we want to query the database and
	the `find` parameter will be used as the AQL query.

	When storing the dataset into the database, the resulting dataset will have
	an additional column added: `country_iso`, which represents the ISO 3166-3
	code of the country, in this case 'USA.

	The dataset will be stored in the `county_health_rankings` collection. The
	list of dataset columns is stored in the `index` collection as follows:

	{
		_key: 'county_health_rankings',
		columns: [<dataset/collection column names>]
	}
	'''

	# Globals
	kCollection_name = 'county_health_rankings'

	###
	# Load from file.
	###
	if file is not None:

		# Load from file
		df = pd.read_csv(file)

		# Add country.
		df['country_iso'] = 'USA'

		# Return DataFrame if password not provided.
		if pswd is None:
			return df													# ==>

		###
		# Store in database
		###
		client = ArangoClient(hosts=host)
		db = client.db(base, username=user, password=pswd)
		if db.has_collection(kCollection_name):
			collection = db.collection(kCollection_name)
		else:
			collection = db.create_collection(kCollection_name, edge=False)
		if db.has_collection('index'):
			index = db.collection('index')
		else:
			index = db.create_collection('index', edge=False)

		# Load dataset columns.
		if index.has(kCollection_name):
			index.update(dict(
				_key=kCollection_name,
				columns=list(df.columns)
			))
		else:
			index.insert(dict(
				_key=kCollection_name,
				columns=list(df.columns)
			))

		# Load dataset.
		collection.truncate()
		records = [
				{k:v for k,v in m.items() if pd.notnull(v)}
				for m in df.to_dict(orient='rows')
			]
		for record in records:
			result = collection.insert(record, sync=True)
			break
		return result
		# return collection.insert_many(
		# 	[
		# 		{k:v for k,v in m.items() if pd.notnull(v)}
		# 		for m in df.to_dict(orient='rows')
		# 	],
		# 	sync=True
		# )

		return df.drop('_key', axis=1)									# ==>

	###
	# Retrieve from database
	###
	client = ArangoClient(hosts=host)
	db = client.db(base, username=user, password=pswd)
	cursor = db.aql.execute(find)

	# Return DataFrame.
	return pd.DataFrame.from_records(
		[document for document in cursor],
		exclude=['_key', '_id', '_rev']
	)  																	# ==>
