################################################################################
# Database management class.                                                   #
#                                                                              #
# Manage input/output operations with database.                                #
################################################################################

# Import libraries.
import re
import numpy as np
import pandas as pd
from arango import ArangoClient


class Database:

	# Static members
	chunk_size_ = 1000
	index_name_ = 'index'
	pat_sep = re.compile('[,][ ]?')
	pat_grp = re.compile('(((([-]?\d+\.\d+) ([-]?\d+\.\d+))[, ]?)+[, ]?)+')
	known_datasets_ = [
		'coders_against_covid', 'county_health_rankings',
		'canada_open_data_working_group', 'covid_tracker_canada',
		'covid_sources_for_counties', 'covid_sources_for_states',
		'covid_statistics_for_states_daily', 'ecdc_worldwide',
		'cdcs_social_vulnerability_index_tract_level',
		'cdcs_social_vulnerability_index_county_level',
		'cdphe_health_facilities', 'coronavirus_world_airport_impacts',
		'definitive_healthcare_usa_hospital_beds'
	]
	us_states = {
		'Alabama': 'AL',
		'Alaska': 'AK',
		'Arizona': 'AZ',
		'Arkansas': 'AR',
		'California': 'CA',
		'Colorado': 'CO',
		'Connecticut': 'CT',
		'Delaware': 'DE',
		'District of Columbia': 'DC',
		'Florida': 'FL',
		'Georgia': 'GA',
		'Hawaii': 'HI',
		'Idaho': 'ID',
		'Illinois': 'IL',
		'Indiana': 'IN',
		'Iowa': 'IA',
		'Kansas': 'KS',
		'Kentucky': 'KY',
		'Louisiana': 'LA',
		'Maine': 'ME',
		'Maryland': 'MD',
		'Massachusetts': 'MA',
		'Michigan': 'MI',
		'Minnesota': 'MN',
		'Mississippi': 'MS',
		'Missouri': 'MO',
		'Montana': 'MT',
		'Nebraska': 'NE',
		'Nevada': 'NV',
		'New Hampshire': 'NH',
		'New Jersey': 'NJ',
		'New Mexico': 'NM',
		'New York': 'NY',
		'North Carolina': 'NC',
		'North Dakota': 'ND',
		'Ohio': 'OH',
		'Oklahoma': 'OK',
		'Oregon': 'OR',
		'Pennsylvania': 'PA',
		'Rhode Island': 'RI',
		'South Carolina': 'SC',
		'South Dakota': 'SD',
		'Tennessee': 'TN',
		'Texas': 'TX',
		'Utah': 'UT',
		'Vermont': 'VT',
		'Virginia': 'VA',
		'Washington': 'WA',
		'West Virginia': 'WV',
		'Wisconsin': 'WI',
		'Wyoming': 'WY',
		'American Samoa': 'AS',
		'Guam': 'GU',
		'Marshall Islands': 'MH',
		'Micronesia': 'FM',
		'Northern Marianas': 'MP',
		'Palau': 'PW',
		'Puerto Rico': 'PR',
		'Virgin Islands': 'VI'
	}
	cn_states = {
		'Newfoundland and Labrador': 'NL',
		'Prince Edward Island': 'PE',
		'Nova Scotia': 'NS',
		'New Brunswick': 'NB',
		'Quebec': 'QC',
		'Ontario': 'ON',
		'Manitoba': 'MB',
		'Saskatchewan': 'SK',
		'Alberta': 'AB',
		'British Columbia': 'BC',
		'Yukon': 'YT',
		'Northwest Territories': 'NT',
		'Nunavut': 'NU'
	}

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
				columns = set()

				# Handle existing columns.
				if not do_clear:
					if self.ix.has(name):
						temp = collection.get(name)
						if temp is not None:
							columns = set(temp['columns'])

				# Update columns
				columns.update(df.columns)

				# Has index
				if self.ix.has(name):
					self.ix.update(dict(
						_key=name,
						file=file,
						collection=col_name,
						columns=list(list(columns))
					))

				# Has no index
				else:
					self.ix.insert(dict(
						_key=name,
						file=file,
						collection=col_name,
						columns=list(list(columns))
					))

				done_index = True

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

	# Parse geometries
	def parse_geometries(self, geometry):
		'''
		Process the provided geometry and return in GeoJSON format.
		Supports POINT and POLYGON

		:param geometry: str, The geometry in the dataset.
		:return: dict, GeoJSON geometry.
		'''

		# Get type
		parts = geometry.split(' ', 1)

		# Parse Points
		if parts[0] == 'POINT':
			return dict(
				type = 'Point',
				coordinates = list(
					np.array(
						re.search(self.pat_grp, parts[1]).group(0).split()
					).astype(np.float)
				)
			)

		# Parse polygons
		elif parts[0] == 'POLYGON':
			return dict(
				type = 'Polygon',
				coordinates = [
					[
						list(
							np.array(y.split()).astype(np.float)
						)
						for y in x
					]
					for x in
					[
						re.split(self.pat_sep, y) for y in
						[
							x.group(0) for x in
							re.finditer(
								self.pat_grp,
								parts[1]
							)
						]
					]
				]
			)

		# Fail on other shapes
		else:
			raise Exception("Unsupported shape: {}".format(parts[0]))

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
		elif name == 'cdcs_social_vulnerability_index_tract_level':
			return 'cdcs_social_vulnerability_index_tract'
		elif name == 'cdcs_social_vulnerability_index_county_level':
			return 'cdcs_social_vulnerability_index_county'
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
		elif name == 'ecdc_worldwide':
			self.process_ecdc_worldwide(dataset)
		elif name == 'cdcs_social_vulnerability_index_tract_level':
			self.process_cdcs_social_vulnerability_index_tract_level(dataset)
		elif name == 'cdcs_social_vulnerability_index_county_level':
			self.process_cdcs_social_vulnerability_index_county_level(dataset)
		elif name == 'cdphe_health_facilities':
			self.process_cdphe_health_facilities(dataset)
		elif name == 'coronavirus_world_airport_impacts':
			self.process_coronavirus_world_airport_impacts(dataset)
		elif name == 'definitive_healthcare_usa_hospital_beds':
			self.process_definitive_healthcare_usa_hospital_beds(dataset)

	# Process 'coders_against_covid' dataset
	def process_coders_against_covid(self, dataset):
		'''
		Process t/f fields into booleans, normalise `geometry` field
		into GeoJSON format and add `iso_level_1` field ('USA').

		:param dataset: Dataset to process
		'''

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

		# Normalise geometry field.
		dataset['geometry'] = dataset['geometry'].apply(
			lambda x:
				None if pd.isna(x)
				else self.parse_geometries(x)
		)

		# Indicate has shape
		dataset['_dataset_has_shape'] = True

		# Add country.
		dataset['iso_level_1'] = 'USA'

		# Copy to iso level 2
		dataset['iso_level_2'] = \
			dataset['location_address_region'].apply(
				lambda x:
					'USA-{}'.format(self.us_states[x]) if x in self.us_states.keys()
					else x
			)

		# Copy to iso level 3
		dataset['iso_level_3'] = dataset['location_address_locality']

	# Process 'county_health_rankings' dataset
	def process_county_health_rankings(self, dataset):
		'''
		Add `iso_level_1` field ('USA').

		:param dataset: Dataset to process
		'''

		# Add country.
		dataset['iso_level_1'] = 'USA'

		# Add state
		dataset['iso_level_2'] = \
			dataset['state'].apply(
				lambda x:
					'USA-{}'.format(self.us_states[x]) if x in self.us_states.keys()
					else x
			)

		# Add county
		dataset['iso_level_3'] = dataset['county']

	# Process 'canada_open_data_working_group' dataset
	def process_canada_open_data_working_group(self, dataset):
		'''
		Drop `case_id field, convert `sex` into `sex_male` boolean, convert
		`has_travel_history` to boolean and add `iso_level_1` field ('CAN').

		:param dataset: Dataset to process
		'''

		# Drop case_id
		dataset.drop('case_id', axis=1, inplace=True)

		# Indicate has shape
		dataset['_dataset_has_shape'] = False

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

		# Add country.
		dataset['iso_level_1'] = 'CAN'

		# Add province
		dataset['iso_level_2'] = \
			dataset['province'].apply(
				lambda x:
					'CAN-{}'.format(self.cn_states[x]) if x in self.us_states.keys()
					else x
			)

		# Add county
		dataset['iso_level_3'] = dataset['health_region']

	# Process 'covid_tracker_canada' dataset
	def process_covid_tracker_canada(self, dataset):
		'''
		Set _key to `id` field, convert `confirmed_presumptive` field to
		boolean and add `iso_level_1` field ('CAN').

		:param dataset: Dataset to process
		'''

		# Set record key
		dataset['_key'] = dataset['id'].apply(str)

		# Normalise confirmed_presumptive
		dataset['confirmed_presumptive'] = dataset['confirmed_presumptive'] \
			.apply(lambda x: True if x == 'CONFIRMED' else False)

		# Indicate has shape
		dataset['_dataset_has_shape'] = False

		# Add country.
		dataset['iso_level_1'] = 'CAN'

		# Add province
		dataset['iso_level_2'] = \
			dataset['province'].apply(
				lambda x:
					'CAN-{}'.format(self.cn_states[x]) if x in self.us_states.keys()
					else x
			)

		# Add county
		dataset['iso_level_3'] = dataset['city']

	# Process 'covid_sources_for_counties' dataset
	def process_covid_sources_for_counties(self, dataset):
		'''
		Set key to iso_level_1 + `state` + `county` and add `iso_level_1` field
		('USA').

		:param dataset: Dataset to process
		'''

		# Set record key
		dataset['_key'] = dataset.apply(
			lambda x: '-'.join(
				[
					'USA',
					x['state'],
					x['county'].replace(' ', '_')
				]
			),
			axis=1
		)

		# Indicate has shape
		dataset['_dataset_has_shape'] = False

		# Add country.
		dataset['iso_level_1'] = 'USA'

		# Add province
		dataset['iso_level_2'] = \
			dataset['state'].apply(
				lambda x:
					'USA-{}'.format(self.us_states[x]) if x in self.us_states.keys()
					else x
			)

		# Add county
		dataset['iso_level_3'] = dataset['county']

	# Process 'covid_sources_for_states' dataset
	def process_covid_sources_for_states(self, dataset):
		'''
		Set key to iso_level_1 + `state`, set `pum` to boolean and add `iso_level_1`
		field ('USA').

		:param dataset: Dataset to process
		'''

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
					'USA',
					x['state']
				]
			),
			axis=1
		)

		# Indicate has shape
		dataset['_dataset_has_shape'] = False

		# Add country.
		dataset['iso_level_1'] = 'USA'

		# Add province
		dataset['iso_level_2'] = \
			dataset['state'].apply(
				lambda x:
					'USA-{}'.format(self.us_states[x]) if x in self.us_states.keys()
					else x
			)

	# Process 'covid_statistics_for_states_daily' dataset
	def process_covid_statistics_for_states_daily(self, dataset):
		'''
		Set key field to `hash` and add `iso_level_1` field ('USA').

		:param dataset: Dataset to process
		'''

		# Set record key
		dataset['_key'] = dataset['hash']

		# Indicate has shape
		dataset['_dataset_has_shape'] = False

		# Add country.
		dataset['iso_level_1'] = 'USA'

		# Add province
		dataset['iso_level_2'] = \
			dataset['state'].apply(
				lambda x:
					'USA-{}'.format(self.us_states[x]) if x in self.us_states.keys()
					else x
			)

	# Process 'ecdc_worldwide' dataset
	def process_ecdc_worldwide(self, dataset):
		'''
		Set `iso_level_1` field to `countryterritorycode`.

		:param dataset: Dataset to process
		'''

		# Indicate has shape
		dataset['_dataset_has_shape'] = False

		# Add country.
		dataset['iso_level_1'] = dataset['countryterritorycode']

	# Process 'cdcs_social_vulnerability_index_tract_level' dataset
	def process_cdcs_social_vulnerability_index_tract_level(self, dataset):
		'''
		Set `iso_level_1` to 'USA',
		`iso_level_2` to `st_abbr`, iso_level_3` to `county`, normalise t/f
		boolean fields and convert geometry to GeoJSON format.

		:param dataset: Dataset to process
		'''

		# Normalise boolean fields.
		bool_fields = [
			'f_pov', 'f_nohsdp', 'f_age65', 'f_age17', 'f_disabl',
			'f_sngpnt','f_minrty', 'f_limeng', 'f_munit', 'f_mobile',
			'f_crowd', 'f_noveh', 'f_groupq'
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

		# Normalise GeoJSON geometry
		dataset['geometry'] = dataset['geometry'].apply(
			lambda x:
				None if pd.isna(x)
				else self.parse_geometries(x)
		)

		# Indicate has shape
		dataset['_dataset_has_shape'] = True

		# Add country.
		dataset['iso_level_1'] = 'USA'

		# Add province
		dataset['iso_level_2'] = \
			dataset['st_abbr'].apply(
				lambda x:
					'USA-{}'.format(self.us_states[x]) if x in self.us_states.keys()
					else x
			)

		# Add county
		dataset['iso_level_3'] = dataset['county']

	# Process 'cdcs_social_vulnerability_index_county_level' dataset
	def process_cdcs_social_vulnerability_index_county_level(self, dataset):
		'''
		Set `iso_level_1` to 'USA',
		`iso_level_2` to `st_abbr`, iso_level_3` to `county`, normalise t/f
		boolean fields and convert geometry to GeoJSON format.

		:param dataset: Dataset to process
		'''

		# Normalise boolean fields.
		bool_fields = [
			'f_pov', 'f_unemp', 'f_pci', 'f_nohsdp', 'f_theme1',
			'f_age65', 'f_age17', 'f_disabl', 'f_sngpnt', 'f_theme2',
			'f_minrty', 'f_limeng', 'f_theme3', 'f_munit', 'f_mobile',
			'f_crowd', 'f_noveh', 'f_groupq', 'f_theme4', 'f_total'
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

		# Normalise GeoJSON geometry
		dataset['geometry'] = dataset['geometry'].apply(
			lambda x:
				None if pd.isna(x)
				else self.parse_geometries(x)
		)

		# Indicate has shape
		dataset['_dataset_has_shape'] = True

		# Add country.
		dataset['iso_level_1'] = 'USA'

		# Add province
		dataset['iso_level_2'] = \
			dataset['st_abbr'].apply(
				lambda x:
					'USA-{}'.format(self.us_states[x]) if x in self.us_states.keys()
					else x
			)

		# Add county
		dataset['iso_level_3'] = dataset['county']

	# Process 'cdphe_health_facilities' dataset
	def process_cdphe_health_facilities(self, dataset):
		'''
		Process t/f fields into booleans, normalise `geometry` field
		into GeoJSON format and add ISO fields.

		:param dataset: Dataset to process
		'''

		# Normalise boolean fields.
		bool_fields = [
			'medicare', 'medicaid'
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

		# Normalise geometry field.
		dataset['geometry'] = dataset['geometry'].apply(
			lambda x:
				None if pd.isna(x)
				else self.parse_geometries(x)
		)

		# Indicate has shape
		dataset['_dataset_has_shape'] = True

		# Add country.
		dataset['iso_level_1'] = 'USA'

		# Copy to iso level 2
		dataset['iso_level_2'] = \
			dataset['state'].apply(
				lambda x:
					'USA-{}'.format(self.us_states[x]) if x in self.us_states.keys()
					else x
			)

		# Copy to iso level 3
		dataset['iso_level_3'] = dataset['county']

		# Copy to iso level 4
		dataset['iso_level_4'] = dataset['city']

	# Process 'coronavirus_world_airport_impacts' dataset
	def process_coronavirus_world_airport_impacts(self, dataset):
		'''
		Process t/f fields into booleans, normalise `geometry` field
		into GeoJSON format, set key field to `ident` and add ISO fields.

		:param dataset: Dataset to process
		'''

		# Normalise boolean fields.
		bool_fields = [
			'scheduled'
		]

		for field in bool_fields:
			dataset[field] = dataset[field].apply(
				lambda x:
					True if x == 'yes'
					else (
						False if x == 'no'
						else None
					)
			)

		# Normalise geometry field.
		dataset['geometry'] = dataset['geometry'].apply(
			lambda x:
				None if pd.isna(x)
				else self.parse_geometries(x)
		)

		# Indicate has shape
		dataset['_dataset_has_shape'] = True

		# Add country.
		dataset['iso_level_1'] = dataset['iso_countr']

		# Copy to iso level 2
		dataset['iso_level_2'] = dataset['iso_region']

		# Copy to iso level 3
		dataset['iso_level_3'] = dataset['municipali']

	# Process 'definitive_healthcare_usa_hospital_beds' dataset
	def process_definitive_healthcare_usa_hospital_beds(self, dataset):
		'''
		Normalise `geometry` field
		into GeoJSON format and add ISO fields.

		:param dataset: Dataset to process
		'''

		# Normalise geometry field.
		dataset['geometry'] = dataset['geometry'].apply(
			lambda x:
				None if pd.isna(x)
				else self.parse_geometries(x)
		)

		# Indicate has shape
		dataset['_dataset_has_shape'] = True

		# Add country.
		dataset['iso_level_1'] = 'USA'

		# Copy to iso level 2
		dataset['iso_level_2'] = \
			dataset['hq_state'].apply(
				lambda x:
					'USA-{}'.format(self.us_states[x]) if x in self.us_states.keys()
					else x
			)

		# Copy to iso level 3
		dataset['iso_level_3'] = dataset['county_nam']

		# Copy to iso level 4
		dataset['iso_level_4'] = dataset['hq_city']
