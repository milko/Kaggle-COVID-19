################################################################################
# Database management class.                                                   #
#                                                                              #
# Manage input/output operations with database.                                #
################################################################################

# Import libraries.
import re
import pandas as pd
from arango import ArangoClient


class Database:

	# Static members
	chunk_size_ = 1000
	index_name_ = 'index'
	known_datasets_ = [
		'coders_against_covid', 'county_health_rankings',
		'canada_open_data_working_group', 'covid_tracker_canada',
		'covid_sources_for_counties', 'covid_sources_for_states',
		'covid_statistics_for_states_daily', 'ecdc_worldwide'
	]

	# Instantiate object
	def __init__(
			self,
			pswd,
			host='http://localhost:8529',
			base='Kaggle-Covid-19',
			user='root'):
		'''
		Instantiate database with endpoint, database name, user name and
		password.

		:param host: Database host, defaults to 'http://localhost:8529'.
		:param base: Database name, defaults to 'Kaggle-Covid-19'.
		:param user: Database user name, defaults to 'root.
		:param pswd: Database user password, must be provided
		'''

		# Load data members
		self.cl = ArangoClient(hosts=host)
		self.db = self.cl.db(base, username=user, password=pswd)

		# Create index collection
		if self.db.has_collection(self.index_name_):
			self.ix = self.db.collection(self.index_name_)
		else:
			self.ix = self.db.create_collection(self.index_name_, edge=False)

	# Load file
	def load_file(self, file, name, is_edge=False, do_clear=True):
		'''
		Load provided file into database.

		:param file: File path to dataset.
		:param name: Dataset name, becomes collection name.
		:return: int, number of rows.
		'''

		# Get/create collection
		col_name = self.select_collection(name)
		if self.db.has_collection(col_name):
			collection = self.db.collection(col_name)
		else:
			collection = self.db.create_collection(col_name, edge=is_edge)

		# Truncate collection
		if do_clear:
			collection.truncate()

		# Load data
		done_index = False
		reader = pd.read_csv(file, chunksize=self.chunk_size_)
		for df in reader:

			# Save columns to index.
			if not done_index:
				if self.ix.has(name):
					self.ix.update(dict(
						_key=name,
						file=file,
						collection=col_name,
						columns=list(df.columns)
					))
				else:
					self.ix.insert(dict(
						_key=name,
						file=file,
						collection=col_name,
						columns=list(df.columns)
					))

			# Process the dataset
			if name in self.known_datasets_:
				self.process_dataset(df, name)

			# Load the data
			collection.import_bulk(
				[
					{k: v for k, v in m.items() if pd.notnull(v)}
					for m in df.to_dict(orient='rows')
				],
				on_duplicate='error',
				sync=True
			)

		# Return record count.
		return collection.count()										# ==>

	# Select collection
	def select_collection(self, name):
		'''
		Return collection name according to dataset name.

		:param name: Dataset name
		:return: str, collection name.
		'''

		# Parse by name
		if name == 'covid_sources_for_counties':
			return 'covid_tracking_project_sources'
		elif name == 'covid_sources_for_states':
			return 'covid_tracking_project_sources'
		elif name == 'covid_statistics_for_states_daily':
			return 'covid_tracking_project_data'
		else:
			return name													# ==>

	# Process dataset
	def process_dataset(self, dataset, name):
		'''
		Process dataset before storing in database.

		:param dataset: Pandas DataFrame
		:param name: Dataset name/collection name
		'''

		# Parse by name
		if name == 'coders_against_covid':
			self.process_coders_against_covid(dataset)
		elif name == 'county_health_rankings':
			self.process_county_health_rankings(dataset)
		elif name == 'canada_open_data_working_group':
			self.process_canada_open_data_working_group(dataset)
		elif name == 'covid_tracker_canada':
			self.process_covid_tracker_canada(dataset)
		elif name == 'covid_sources_for_counties':
			self.process_covid_sources_for_counties(dataset)
		elif name == 'covid_sources_for_states':
			self.process_covid_sources_for_states(dataset)
		elif name == 'covid_statistics_for_states_daily':
			self.process_covid_statistics_for_states_daily(dataset)
		elif name == 'ecdc_worldwide':
			self.process_ecdc_worldwide(dataset)

	# Process 'coders_against_covid' dataset
	def process_coders_against_covid(self, dataset):
		'''
		Process t/f fields into booleans, normalise `geometry` field
		into GeoJSON format and add `iso_level_1` field ('USA').

		:param dataset: Dataset to process
		'''

		# Add country.
		dataset['iso_level_1'] = 'USA'

		# Normalise boolean fields.
		bool_fields = [
			'is_verified', 'is_hidden', 'is_location_screening_patients',
			'is_location_collecting_specimens',
			'is_location_accepting_third_party_orders_for_testing',
			'is_location_only_testing_patients_that_meet_criteria',
			'is_location_by_appointment_only', 'is_ready_for_prod'
		]

		for field in bool_fields:
			dataset[field] = dataset[field].apply(
				lambda x:
					True if x == 't'
					else (
						False if x == 'f'
						else None
					)
			)

		# Normalise state field.
		state_names = {
			'Arizona': 'AZ',
			'Arkansas': 'AR',
			'California': 'CA',
			'Colorado': 'CO',
			'Connecticut': 'CT',
			'Florida': 'FL',
			'Hawaii': 'HI',
			'Idaho': 'ID',
			'Indiana': 'IN',
			'Iowa': 'IA',
			'Kansas': 'KS',
			'Kentucky': 'KY',
			'Louisiana': 'LA',
			'Maryland': 'MD',
			'Massachusetts': 'MA',
			'Michigan': 'MI',
			'Minnesota': 'MN',
			'Mississippi': 'MS',
			'Montana': 'MT',
			'Nebraska': 'NE',
			'New Mexico': 'NM',
			'North Carolina': 'NC',
			'North Dakota': 'ND',
			'Ohio': 'OH',
			'Oklahoma': 'OK',
			'Oregon': 'OR',
			'Pennsylvania': 'PA',
			'South Carolina': 'SC',
			'South Dakota': 'SD',
			'Tennessee': 'TN',
			'Utah': 'UT',
			'Vermont': 'VT',
			'Virginia': 'VA',
			'West Virginia': 'WV',
			'Wisconsin': 'WI',
			'Wyoming': 'WY'
		}

		dataset['location_address_region'] = \
			dataset['location_address_region'].apply(
				lambda x: state_names[x]
					if x in state_names.keys()
					else x
			)

		# Copy to iso level 2
		dataset['iso_level_2'] = dataset['location_address_region']

		# Copy to iso level 3
		dataset['iso_level_3'] = \
			dataset['location_address_locality'].replace(' ', '_')

		# Normalise geometry field.
		dataset['geometry'] = dataset['geometry'].apply(
			lambda x:
				None if pd.isna(x)
				else dict(
					type='Point',
					coordinates=[
						float(re.search('POINT \((.+) (.+)\)', x).group(1)),
						float(re.search('POINT \((.+) (.+)\)', x).group(2))
					]
				)
		)

	# Process 'county_health_rankings' dataset
	def process_county_health_rankings(self, dataset):
		'''
		Add `iso_level_1` field ('USA').

		:param dataset: Dataset to process
		'''

		# Add country.
		dataset['iso_level_1'] = 'USA'

	# Process 'canada_open_data_working_group' dataset
	def process_canada_open_data_working_group(self, dataset):
		'''
		Drop `case_id field, convert `sex` into `sex_male` boolean, convert
		`has_travel_history` to boolean and add `iso_level_1` field ('CAN').

		:param dataset: Dataset to process
		'''

		# Add country.
		dataset['iso_level_1'] = 'CAN'

		# Drop case_id
		dataset.drop('case_id', axis=1, inplace=True)

		# Normalise sex
		dataset['sex_male'] = dataset['sex'][dataset['sex'] != 'Not Reported'] \
			.apply(lambda x: True if x == 'Male' else False)

		# Normalise boolean fields.
		bool_fields = [
			'has_travel_history'
		]

		for field in bool_fields:
			dataset[field] = dataset[field].apply(
				lambda x:
					True if x == 't'
					else (
						False if x == 'f'
						else None
					)
			)

	# Process 'covid_tracker_canada' dataset
	def process_covid_tracker_canada(self, dataset):
		'''
		Set _key to `id` field, convert `confirmed_presumptive` field to
		boolean and add `iso_level_1` field ('CAN').

		:param dataset: Dataset to process
		'''

		# Add country.
		dataset['iso_level_1'] = 'CAN'

		# Set record key
		dataset['_key'] = dataset['id'].apply(str)

		# Normalise confirmed_presumptive
		dataset['confirmed_presumptive'] = dataset['confirmed_presumptive'] \
			.apply(lambda x: True if x == 'CONFIRMED' else False)

	# Process 'covid_sources_for_counties' dataset
	def process_covid_sources_for_counties(self, dataset):
		'''
		Set key to iso_level_1 + `state` + `county` and add `iso_level_1` field
		('USA').

		:param dataset: Dataset to process
		'''

		# Add country.
		dataset['iso_level_1'] = 'USA'

		# Set record key
		dataset['_key'] = dataset.apply(
			lambda x: '-'.join(
				[
					x['iso_level_1'],
					x['state'],
					x['county'].replace(' ', '_')
				]
			),
			axis=1
		)

	# Process 'covid_sources_for_states' dataset
	def process_covid_sources_for_states(self, dataset):
		'''
		Set key to iso_level_1 + `state`, set `pum` to boolean and add `iso_level_1`
		field ('USA').

		:param dataset: Dataset to process
		'''

		# Add country.
		dataset['iso_level_1'] = 'USA'

		# Set `pum` to boolean
		dataset['pum'] = dataset['pum'].apply(
			lambda x:
				True if x == 't'
				else (
					False if x == 'f'
					else None
				)
		)

		# Set record key
		dataset['_key'] = dataset.apply(
			lambda x: '-'.join(
				[
					x['iso_level_1'],
					x['state']
				]
			),
			axis=1
		)

	# Process 'covid_statistics_for_states_daily' dataset
	def process_covid_statistics_for_states_daily(self, dataset):
		'''
		Set key field to `hash` and add `iso_level_1` field ('USA').

		:param dataset: Dataset to process
		'''

		# Add country.
		dataset['iso_level_1'] = 'USA'

		# Set record key
		dataset['_key'] = dataset['hash']

	# Process 'ecdc_worldwide' dataset
	def process_ecdc_worldwide(self, dataset):
		'''
		Set `iso_level_1` field to `countryterritorycode`.

		:param dataset: Dataset to process
		'''

		# Add country.
		dataset['iso_level_1'] = dataset['countryterritorycode']
