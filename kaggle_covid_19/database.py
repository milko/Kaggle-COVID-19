################################################################################
# Database management class.                                                   #
#                                                                              #
# Manage input/output operations with database.                                #
################################################################################

# Import libraries.
import re
import os
import glob
import numpy as np
import pandas as pd
from types import *
from arango import ArangoClient


class Database:

	# Static members
	chunk_size_ = 1000
	index_name_ = 'index'
	shape_name_ = 'shapes'
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
		'definitive_healthcare_usa_hospital_beds',
		'border_wait_times_at_us_canada_border',
		'github_belgium_regions', 'github_italy_regions',
		'github_uk_regions', 'github_france_regions',
		'github_spain_regions',
		'harvard_global_health_institute_20',
		'harvard_global_health_institute_40',
		'harvard_global_health_institute_60',
		'hde_acaps_government_measures',
		'hde_global_school_closures',
		'hde_inform_covid_indicators',
		'hde_total_covid_tests',
		'hifld_aircraft_landing_facilities',
		'hifld_hospitals',
		'hifld_local_emergency_operations_centers',
		'hifld_nursing_homes',
		'hifld_public_health_departments',
		'hifld_urgent_care_facilities',
		'hifld_us_ports_of_entry',
		'ihme_hospitalisation',
		'nextstrain_phylogeny',
		'owd_tests_conducted',
		'wfp_travel_restrictions',
		'cdc_cities_census_tract_level',
		'cdc_global_adult_tobacco_survey', 'cdc_global_youth_tobacco_survey',
		'cdc_behavioral_risk_factor_surveillance',
		'cdc_chronic_disease_indicators',
		'usafacts_covid_19_by_county',
		'who_tuberculosis_case_notifications', 'who_situation_reports_covid_19',
		'wb_indicators',
		'wb_global_population',
		'wb_health_population_summary',
		'wb_world_development_indicators_summary'
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
			pswd: str,
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
	def load_file(self, file: str, name: str, is_edge=False, do_clear=True) -> int:
		'''
		Load provided file into database.

		:param file: File path to dataset.
		:param name: Dataset name, becomes collection name.
		:return: int, number of rows.
		'''

		# Intercept border waiting times
		if name == 'border_wait_times_at_us_canada_border':
			return \
				self.process_border_wait_times_at_us_canada_border(file)# ==>

		# Get/create collection
		col_name = self.select_collection(name)
		if self.db.has_collection(col_name):
			collection = self.db.collection(col_name)
		else:
			collection = self.db.create_collection(col_name, edge=is_edge)

		# Truncate collection
		if do_clear:
			collection.truncate()

		# Intercept our world in data datasets
		if name == 'owd_tests_conducted':

			# Merge directory datasets
			df = None
			if file[-1] != '/':
				file += '/'
			for item in glob.glob(file + "*.csv"):
				if df is None:
					df = pd.read_csv(item)
				else:
					df = df.merge(
						pd.read_csv(item),
						how='outer',
						on=['entity', 'code', 'date']
					)

			# Update columns
			columns = set(df.columns)

			# Has index
			if self.ix.has(name):
				self.ix.update(dict(
					_key=name,
					file=file,
					collection=col_name,
					columns=list(list(columns)),
					has_shape=self.dataset_has_shape(name)
				))

			# Has no index
			else:
				self.ix.insert(dict(
					_key=name,
					file=file,
					collection=col_name,
					columns=list(list(columns)),
					has_shape=self.dataset_has_shape(name)
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

		# Intercept USAFacts datasets
		elif name == 'usafacts_covid_19_by_county':

			# Merge directory datasets
			df = None
			if file[-1] != '/':
				file += '/'
			for item in glob.glob(file + "*.csv"):
				if df is None:
					df = pd.read_csv(item)
				else:
					df = df.merge(
						pd.read_csv(item),
						how='outer',
						on=[
							'county_fips', 'county_name', 'state_name',
							'state_fips', 'date', 'lat', 'long', 'geometry'
						]
					)

			# Update columns
			columns = set(df.columns)

			# Has index
			if self.ix.has(name):
				self.ix.update(dict(
					_key=name,
					file=file,
					collection=col_name,
					columns=list(list(columns)),
					has_shape=self.dataset_has_shape(name)
				))

			# Has no index
			else:
				self.ix.insert(dict(
					_key=name,
					file=file,
					collection=col_name,
					columns=list(list(columns)),
					has_shape=self.dataset_has_shape(name)
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

		# Intercept World Bank data
		elif name == 'wb_indicators':

			# Datasets using indicators
			files = [
				'climate-change.csv',
				'community-health-workers-per-1-000-people.csv',
				'environment-social-and-governance-data.csv',
				'hospital-beds-per-1-000-people.csv',
				'nurses-and-midwives-per-1-000-people.csv',
				'people-with-basic-handwashing-facilities-including-soap-and-water-of-population.csv',
				'physicians-per-1-000-people.csv',
				'smoking-prevalence-total-ages-15.csv',
				'specialist-surgical-workforce-per-100-000-population.csv'
			]

			# Get/create collection
			dict_name = 'wb_ddict'
			if self.db.has_collection(dict_name):
				col = self.db.collection(dict_name)
			else:
				col = self.db.create_collection(dict_name, edge=False)

			# Truncate collection
			if do_clear:
				col.truncate()

			# Create data dictionary
			dictionary = dict()
			df = pd.DataFrame()
			if file[-1] != '/':
				file += '/'
			for item in files:
				df = pd.read_csv(
					file + item,
					usecols=['indicator_code', 'indicator_name']
				).drop_duplicates(ignore_index=True)
				for idx in range(0, len(df) - 1):
					dictionary[df.loc[idx]['indicator_code']] = \
						df.loc[idx]['indicator_name']

			# Create records
			records = []
			for key, value in dictionary.items():
				records.append(dict(_key=key, label=value))

			# Write data dictionary
			col.import_bulk(
				records,
				on_duplicate='error',
				sync=True
			)

			# Iterate datasets
			for item in files:

				# Read the data
				df = pd.read_csv(file + item)\
					.drop(columns=['indicator_name', 'untitled_1'], errors='ignore')\
					.melt(id_vars=['country_name', 'country_code', 'indicator_code'])\
					.rename(columns=dict(variable='year'))\
					.dropna(axis=0, how='any')

				# Normalise year
				df['year'] = df['year'].apply(lambda x: int(x))

				# Set country code
				df['iso_level_1'] = df['country_code']

				# Load the data
				collection.import_bulk(
					[
						{k: v for k, v in m.items() if pd.notnull(v)}
						for m in df.to_dict(orient='rows')
					],
					on_duplicate='error',
					sync=True
				)

		# All others
		else:

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
							columns=list(list(columns)),
							has_shape=self.dataset_has_shape(name)
						))

					# Has no index
					else:
						self.ix.insert(dict(
							_key=name,
							file=file,
							collection=col_name,
							columns=list(list(columns)),
							has_shape=self.dataset_has_shape(name)
						))

					done_index = True

				# Process the dataset
				if name in self.known_datasets_:
					df = self.process_dataset(df, name)

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
	def parse_geometries(self, geometry: str):
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
	def select_collection(self, name: str) -> str:
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
		elif name == 'harvard_global_health_institute_20':
			return 'harvard_global_health_institute'
		elif name == 'harvard_global_health_institute_40':
			return 'harvard_global_health_institute'
		elif name == 'harvard_global_health_institute_60':
			return 'harvard_global_health_institute'
		else:
			return name													# ==>

	# Process dataset
	def process_dataset(self, dataset: pd.DataFrame, name: str):
		'''
		Process dataset before storing in database.

		:param dataset: Pandas DataFrame
		:param name: Dataset name/collection name
		'''

		# Parse by name
		if name == 'coders_against_covid':
			dataset = self.process_coders_against_covid(dataset)
		elif name == 'county_health_rankings':
			dataset = self.process_county_health_rankings(dataset)
		elif name == 'canada_open_data_working_group':
			dataset = self.process_canada_open_data_working_group(dataset)
		elif name == 'covid_tracker_canada':
			dataset = self.process_covid_tracker_canada(dataset)
		elif name == 'covid_sources_for_counties':
			dataset = self.process_covid_sources_for_counties(dataset)
		elif name == 'covid_sources_for_states':
			dataset = self.process_covid_sources_for_states(dataset)
		elif name == 'covid_statistics_for_states_daily':
			dataset = self.process_covid_statistics_for_states_daily(dataset)
		elif name == 'ecdc_worldwide':
			dataset = self.process_ecdc_worldwide(dataset)
		elif name == 'cdcs_social_vulnerability_index_tract_level':
			dataset = self.process_cdcs_social_vulnerability_index_tract_level(dataset)
		elif name == 'cdcs_social_vulnerability_index_county_level':
			dataset = self.process_cdcs_social_vulnerability_index_county_level(dataset)
		elif name == 'cdphe_health_facilities':
			dataset = self.process_cdphe_health_facilities(dataset)
		elif name == 'coronavirus_world_airport_impacts':
			dataset = self.process_coronavirus_world_airport_impacts(dataset)
		elif name == 'definitive_healthcare_usa_hospital_beds':
			dataset = self.process_definitive_healthcare_usa_hospital_beds(dataset)
		elif name == 'github_belgium_regions':
			dataset = self.process_github_belgium_regions(dataset)
		elif name == 'github_italy_regions':
			dataset = self.process_github_italy_regions(dataset)
		elif name == 'github_uk_regions':
			dataset = self.process_github_uk_regions(dataset)
		elif name == 'github_france_regions':
			dataset = self.process_github_france_regions(dataset)
		elif name == 'github_spain_regions':
			dataset = self.process_github_spain_regions(dataset)
		elif name == 'harvard_global_health_institute_20':
			dataset = self.process_harvard_global_health_institute_20(dataset)
		elif name == 'harvard_global_health_institute_40':
			dataset = self.process_harvard_global_health_institute_40(dataset)
		elif name == 'harvard_global_health_institute_60':
			dataset = self.process_harvard_global_health_institute_60(dataset)
		elif name == 'hde_acaps_government_measures':
			dataset = self.process_hde_acaps_government_measures(dataset)
		elif name == 'hde_global_school_closures':
			dataset = self.process_hde_global_school_closures(dataset)
		elif name == 'hde_inform_covid_indicators':
			dataset = self.process_hde_inform_covid_indicators(dataset)
		elif name == 'hde_total_covid_tests':
			dataset = self.process_hde_total_covid_tests(dataset)
		elif name == 'hifld_aircraft_landing_facilities':
			dataset = self.process_hifld_aircraft_landing_facilities(dataset)
		elif name == 'hifld_hospitals':
			dataset = self.process_hifld_hospitals(dataset)
		elif name == 'hifld_local_emergency_operations_centers':
			dataset = self.process_hifld_local_emergency_operations_centers(dataset)
		elif name == 'hifld_nursing_homes':
			dataset = self.process_hifld_nursing_homes(dataset)
		elif name == 'hifld_public_health_departments':
			dataset = self.process_hifld_public_health_departments(dataset)
		elif name == 'hifld_urgent_care_facilities':
			dataset = self.process_hifld_urgent_care_facilities(dataset)
		elif name == 'hifld_us_ports_of_entry':
			dataset = self.process_hifld_us_ports_of_entry(dataset)
		elif name == 'ihme_hospitalisation':
			dataset = self.process_ihme_hospitalisation(dataset)
		elif name == 'nextstrain_phylogeny':
			dataset = self.process_nextstrain_phylogeny(dataset)
		elif name == 'owd_tests_conducted':
			dataset = self.process_owd_tests_conducted(dataset)
		elif name == 'wfp_travel_restrictions':
			dataset = self.process_wfp_travel_restrictions(dataset)
		elif name == 'cdc_cities_census_tract_level':
			dataset = self.process_cdc_cities_census_tract_level(dataset)
		elif name == 'cdc_global_adult_tobacco_survey':
			dataset = self.process_cdc_global_adult_tobacco_survey(dataset)
		elif name == 'cdc_global_youth_tobacco_survey':
			dataset = self.process_cdc_global_youth_tobacco_survey(dataset)
		elif name == 'cdc_behavioral_risk_factor_surveillance':
			dataset = self.process_cdc_behavioral_risk_factor_surveillance(dataset)
		elif name == 'cdc_chronic_disease_indicators':
			dataset = self.process_cdc_chronic_disease_indicators(dataset)
		elif name == 'usafacts_covid_19_by_county':
			dataset = self.process_usafacts_covid_19_by_county(dataset)
		elif name == 'who_tuberculosis_case_notifications':
			dataset = self.process_who_tuberculosis_case_notifications(dataset)
		elif name == 'wb_global_population':
			dataset = self.process_wb_global_population(dataset)
		elif name == 'wb_health_population_summary':
			dataset = self.process_wb_health_population_summary(dataset)
		elif name == 'wb_world_development_indicators_summary':
			dataset = self.process_wb_world_development_indicators_summary(dataset)

		return dataset													# ==>

	# Check if dataset has a shape
	def dataset_has_shape(self, name: str) -> bool:
		'''
		Check if dataset has shape.

		:param dataset: Pandas DataFrame
		:param name: Dataset name/collection name
		'''

		# Parse by name
		if name == 'coders_against_covid':
			return True
		elif name == 'county_health_rankings':
			return False
		elif name == 'canada_open_data_working_group':
			return False
		elif name == 'covid_tracker_canada':
			return False
		elif name == 'covid_sources_for_counties':
			return False
		elif name == 'covid_sources_for_states':
			return False
		elif name == 'covid_statistics_for_states_daily':
			return False
		elif name == 'ecdc_worldwide':
			return False
		elif name == 'cdcs_social_vulnerability_index_tract_level':
			return True
		elif name == 'cdcs_social_vulnerability_index_county_level':
			return True
		elif name == 'cdphe_health_facilities':
			return True
		elif name == 'coronavirus_world_airport_impacts':
			return True
		elif name == 'definitive_healthcare_usa_hospital_beds':
			return True
		elif name == 'github_belgium_regions':
			return False
		elif name == 'github_italy_regions':
			return False
		elif name == 'github_uk_regions':
			return False
		elif name == 'github_france_regions':
			return False
		elif name == 'github_spain_regions':
			return False
		elif name == 'harvard_global_health_institute_20':
			return False
		elif name == 'harvard_global_health_institute_40':
			return False
		elif name == 'harvard_global_health_institute_60':
			return False
		elif name == 'hde_acaps_government_measures':
			return False
		elif name == 'hde_global_school_closures':
			return False
		elif name == 'hde_inform_covid_indicators':
			return False
		elif name == 'hde_total_covid_tests':
			return False
		elif name == 'hifld_aircraft_landing_facilities':
			return True
		elif name == 'hifld_hospitals':
			return True
		elif name == 'hifld_local_emergency_operations_centers':
			return True
		elif name == 'hifld_nursing_homes':
			return True
		elif name == 'hifld_public_health_departments':
			return True
		elif name == 'hifld_urgent_care_facilities':
			return True
		elif name == 'hifld_us_ports_of_entry':
			return True
		elif name == 'ihme_hospitalisation':
			return False
		elif name == 'nextstrain_phylogeny':
			return False
		elif name == 'owd_tests_conducted':
			return False
		elif name == 'wfp_travel_restrictions':
			return False
		elif name == 'cdc_cities_census_tract_level':
			return False
		elif name == 'cdc_global_adult_tobacco_survey':
			return False
		elif name == 'cdc_global_youth_tobacco_survey':
			return False
		elif name == 'cdc_behavioral_risk_factor_surveillance':
			return False
		elif name == 'cdc_chronic_disease_indicators':
			return False
		elif name == 'usafacts_covid_19_by_county':
			return True
		elif name == 'who_situation_reports_covid_19':
			return False
		elif name == 'wb_indicators':
			return False
		elif name == 'wb_global_population':
			return False
		elif name == 'wb_health_population_summary':
			return False

	# Process 'coders_against_covid' dataset
	def process_coders_against_covid(self, dataset: pd.DataFrame):
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
		dataset['_dataset_item_type'] = \
			dataset['location_place_of_service_type'].apply(
				lambda x:
					None if x is None
					else x.upper()
		)

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

		return dataset  												# ==>

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

		return dataset  												# ==>

	# Process 'canada_open_data_working_group' dataset
	def process_canada_open_data_working_group(self, dataset: pd.DataFrame):
		'''
		Drop `case_id field, convert `sex` into `sex_male` boolean, convert
		`has_travel_history` to boolean and add `iso_level_1` field ('CAN').

		:param dataset: Dataset to process
		'''

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

		return dataset  												# ==>

	# Process 'covid_tracker_canada' dataset
	def process_covid_tracker_canada(self, dataset: pd.DataFrame):
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

		return dataset  												# ==>

	# Process 'covid_sources_for_counties' dataset
	def process_covid_sources_for_counties(self, dataset: pd.DataFrame):
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

		return dataset  												# ==>

	# Process 'covid_sources_for_states' dataset
	def process_covid_sources_for_states(self, dataset: pd.DataFrame):
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

		# Add country.
		dataset['iso_level_1'] = 'USA'

		# Add province
		dataset['iso_level_2'] = \
			dataset['state'].apply(
				lambda x:
					'USA-{}'.format(self.us_states[x]) if x in self.us_states.keys()
					else x
			)

		return dataset  												# ==>

	# Process 'covid_statistics_for_states_daily' dataset
	def process_covid_statistics_for_states_daily(self, dataset: pd.DataFrame):
		'''
		Set key field to `hash` and add `iso_level_1` field ('USA').

		:param dataset: Dataset to process
		'''

		# Set record key
		dataset['_key'] = dataset['hash']

		# Add country.
		dataset['iso_level_1'] = 'USA'

		# Add province
		dataset['iso_level_2'] = \
			dataset['state'].apply(
				lambda x:
					'USA-{}'.format(self.us_states[x]) if x in self.us_states.keys()
					else x
			)

		return dataset  												# ==>

	# Process 'ecdc_worldwide' dataset
	def process_ecdc_worldwide(self, dataset: pd.DataFrame):
		'''
		Set `iso_level_1` field to `countryterritorycode`.

		:param dataset: Dataset to process
		'''

		# Add country.
		dataset['iso_level_1'] = dataset['countryterritorycode']

		return dataset  												# ==>

	# Process 'cdcs_social_vulnerability_index_tract_level' dataset
	def process_cdcs_social_vulnerability_index_tract_level(self, dataset: pd.DataFrame):
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
		dataset['_dataset_item_type'] = 'TRACT'

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

		return dataset  												# ==>

	# Process 'cdcs_social_vulnerability_index_county_level' dataset
	def process_cdcs_social_vulnerability_index_county_level(self, dataset: pd.DataFrame):
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
		dataset['_dataset_item_type'] = 'COUNTY'

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

		return dataset  												# ==>

	# Process 'cdphe_health_facilities' dataset
	def process_cdphe_health_facilities(self, dataset: pd.DataFrame):
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
		dataset['_dataset_item_type'] = 'HEALTH FACILITY'

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

		return dataset  												# ==>

	# Process 'coronavirus_world_airport_impacts' dataset
	def process_coronavirus_world_airport_impacts(self, dataset: pd.DataFrame):
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
		dataset['_dataset_item_type'] = 'AIRPORT'

		# Add country.
		dataset['iso_level_1'] = dataset['iso_countr']

		# Copy to iso level 2
		dataset['iso_level_2'] = dataset['iso_region']

		# Copy to iso level 3
		dataset['iso_level_3'] = dataset['municipali']

		return dataset  												# ==>

	# Process 'definitive_healthcare_usa_hospital_beds' dataset
	def process_definitive_healthcare_usa_hospital_beds(self, dataset: pd.DataFrame):
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
		dataset['_dataset_item_type'] = 'HOSPITAL'

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

		return dataset  												# ==>

	# Process 'github_belgium_regions' dataset
	def process_github_belgium_regions(self, dataset: pd.DataFrame):
		'''
		Load as-is.

		:param dataset: Dataset to process
		'''

		return dataset  												# ==>

	# Process 'github_italy_regions' dataset
	def process_github_italy_regions(self, dataset: pd.DataFrame):
		'''
		Load as-is.

		:param dataset: Dataset to process
		'''

		return dataset  												# ==>

	# Process 'github_uk_regions' dataset
	def process_github_uk_regions(self, dataset: pd.DataFrame):
		'''
		Load as-is.

		:param dataset: Dataset to process
		'''

		return dataset  												# ==>

	# Process 'github_france_regions' dataset
	def process_github_france_regions(self, dataset: pd.DataFrame):
		'''
		Load as-is.

		:param dataset: Dataset to process
		'''

		return dataset  												# ==>

	# Process 'github_spain_regions' dataset
	def process_github_spain_regions(self, dataset: pd.DataFrame):
		'''
		Load as-is.

		:param dataset: Dataset to process
		'''

		return dataset  												# ==>

	# Process 'harvard_global_health_institute_20' dataset
	def process_harvard_global_health_institute_20(self, dataset: pd.DataFrame):
		'''
		Load as-is and add by population contracted.

		:param dataset: Dataset to process
		'''

		# Add by population contracted
		dataset['by_population_contracted'] = 20

		return dataset  												# ==>

	# Process 'harvard_global_health_institute_40' dataset
	def process_harvard_global_health_institute_40(self, dataset: pd.DataFrame):
		'''
		Load as-is and add by population contracted.

		:param dataset: Dataset to process
		'''

		# Add by population contracted
		dataset['by_population_contracted'] = 40

		return dataset  												# ==>

	# Process 'harvard_global_health_institute_60' dataset
	def process_harvard_global_health_institute_60(self, dataset: pd.DataFrame):
		'''
		Load as-is and add by population contracted.

		:param dataset: Dataset to process
		'''

		# Add by population contracted
		dataset['by_population_contracted'] = 60

		return dataset  												# ==>

	# Process 'hde_acaps_government_measures' dataset
	def process_hde_acaps_government_measures(self, dataset: pd.DataFrame):
		'''
		Load as-is and normalise booleans.

		:param dataset: Dataset to process
		'''

		# Normalise boolean fields.
		bool_fields = [
			'targeted_pop_group'
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

		return dataset 													# ==>

	# Process 'hde_global_school_closures' dataset
	def process_hde_global_school_closures(self, dataset: pd.DataFrame):
		'''
		Load as-is.

		:param dataset: Dataset to process
		'''

		return dataset  												# ==>

	# Process 'hde_inform_covid_indicators' dataset
	def process_hde_inform_covid_indicators(self, dataset: pd.DataFrame):
		'''
		Load as-is.

		:param dataset: Dataset to process
		'''

		return dataset  												# ==>

	# Process 'hde_total_covid_tests' dataset
	def process_hde_total_covid_tests(self, dataset: pd.DataFrame):
		'''
		Load as-is.

		:param dataset: Dataset to process
		'''

		return dataset  												# ==>

	# Process 'hifld_aircraft_landing_facilities' dataset
	def process_hifld_aircraft_landing_facilities(self, dataset: pd.DataFrame):
		'''
		Process t/f fields into booleans, normalise `geometry` field
		into GeoJSON format and add ISO fields.

		:param dataset: Dataset to process
		'''

		# Set not available
		dataset.replace(
			to_replace='NOT AVAILABLE',
			value=float('NaN'),
			inplace=True
		)

		# Normalise boolean fields.
		bool_fields = [
			'tieinfss', 'notamservi', 'customsair', 'customslan',
			'militaryjo', 'militaryla', 'atct', 'noncommerc',
			'medicaluse'
		]

		for field in bool_fields:
			dataset[field] = dataset[field].apply(
				lambda x:
					True if x == 't' or x == 'Y' or x == 'YES'
					else (
						False if x == 'f' or x == 'N' or x == 'NO'
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
		dataset['_dataset_item_type'] = dataset['type'].str.upper()

		# Add country.
		dataset['iso_level_1'] = 'USA'

		# Copy to iso level 2
		dataset['iso_level_2'] = dataset['state']

		# Copy to iso level 3
		dataset['iso_level_3'] = dataset['county']

		# Copy to iso level 4
		dataset['iso_level_4'] = dataset['city']

		return dataset  												# ==>

	# Process 'hifld_hospitals' dataset
	def process_hifld_hospitals(self, dataset: pd.DataFrame):
		'''
		Process t/f fields into booleans, normalise `geometry` field
		into GeoJSON format and add ISO fields.

		:param dataset: Dataset to process
		'''

		# Set not available
		dataset.replace(
			to_replace='NOT AVAILABLE',
			value=float('NaN'),
			inplace=True
		)

		# Normalise boolean fields.
		bool_fields = [
			'helipad'
		]

		for field in bool_fields:
			dataset[field] = dataset[field].apply(
				lambda x:
					True if x == 't' or x == 'Y' or x == 'YES' or x == 'H'
					else (
						False if x == 'f' or x == 'N' or x == 'NO'
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
		dataset['_dataset_item_type'] = dataset['type'].str.upper()

		# Add country.
		dataset['iso_level_1'] = dataset['country']

		# Copy to iso level 2
		dataset['iso_level_2'] = dataset['state']

		# Copy to iso level 3
		dataset['iso_level_3'] = dataset['county']

		# Copy to iso level 4
		dataset['iso_level_4'] = dataset['city']

		return dataset  												# ==>

	# Process 'hifld_local_emergency_operations_centers' dataset
	def process_hifld_local_emergency_operations_centers(self, dataset: pd.DataFrame):
		'''
		Process t/f fields into booleans, normalise `geometry` field
		into GeoJSON format and add ISO fields.

		:param dataset: Dataset to process
		'''

		# Set not available
		dataset.replace(
			to_replace='NOT AVAILABLE',
			value=float('NaN'),
			inplace=True
		)

		# Normalise boolean fields.
		bool_fields = [
			'phoneloc', 'generator', 'basement', 'permanent'
		]

		for field in bool_fields:
			dataset[field] = dataset[field].apply(
				lambda x:
					True if x == 't' or x == 'Y' or x == 'YES'
					else (
						False if x == 'f' or x == 'N' or x == 'NO'
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
		dataset['_dataset_item_type'] = 'EMERGENCY OPERATIONS CENTER'

		# Add country.
		dataset['iso_level_1'] = 'USA'

		# Copy to iso level 2
		dataset['iso_level_2'] = dataset['state']

		# Copy to iso level 3
		dataset['iso_level_3'] = dataset['county']

		# Copy to iso level 4
		dataset['iso_level_4'] = dataset['city']

		return dataset  												# ==>

	# Process 'hifld_nursing_homes' dataset
	def process_hifld_nursing_homes(self, dataset: pd.DataFrame):
		'''
		Process t/f fields into booleans, normalise `geometry` field
		into GeoJSON format and add ISO fields.

		:param dataset: Dataset to process
		'''

		# Set not available
		dataset.replace(
			to_replace='NOT AVAILABLE',
			value=float('NaN'),
			inplace=True
		)

		# Normalise geometry field.
		dataset['geometry'] = dataset['geometry'].apply(
			lambda x:
				None if pd.isna(x)
				else self.parse_geometries(x)
		)

		# Indicate has shape
		dataset['_dataset_has_shape'] = True
		dataset['_dataset_item_type'] = dataset['type'].str.upper()

		# Add country.
		dataset['iso_level_1'] = dataset['country']

		# Copy to iso level 2
		dataset['iso_level_2'] = dataset['state']

		# Copy to iso level 3
		dataset['iso_level_3'] = dataset['county']

		# Copy to iso level 4
		dataset['iso_level_4'] = dataset['city']

		return dataset  												# ==>

	# Process 'hifld_public_health_departments' dataset
	def process_hifld_public_health_departments(self, dataset: pd.DataFrame):
		'''
		Process t/f fields into booleans, normalise `geometry` field
		into GeoJSON format and add ISO fields.

		:param dataset: Dataset to process
		'''

		# Set not available
		dataset.replace(
			to_replace='NOT AVAILABLE',
			value=float('NaN'),
			inplace=True
		)

		# Normalise boolean fields.
		bool_fields = [
			'sdr', 'phoneloc'
		]

		for field in bool_fields:
			dataset[field] = dataset[field].apply(
				lambda x:
					True if x == 't' or x == 'Y' or x == 'YES'
					else (
						False if x == 'f' or x == 'N' or x == 'NO'
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
		dataset['_dataset_item_type'] = 'PUBLIC HEALTH DEPARTMENT'

		# Add country.
		dataset['iso_level_1'] = 'USA'

		# Copy to iso level 2
		dataset['iso_level_2'] = dataset['state']

		# Copy to iso level 3
		dataset['iso_level_3'] = dataset['county']

		# Copy to iso level 4
		dataset['iso_level_4'] = dataset['city']

		return dataset  												# ==>

	# Process 'hifld_urgent_care_facilities' dataset
	def process_hifld_urgent_care_facilities(self, dataset: pd.DataFrame):
		'''
		Process t/f fields into booleans, normalise `geometry` field
		into GeoJSON format and add ISO fields.

		:param dataset: Dataset to process
		'''

		# Set not available
		dataset.replace(
			to_replace='NOT AVAILABLE',
			value=float('NaN'),
			inplace=True
		)

		# Normalise boolean fields.
		bool_fields = [
			'phoneloc'
		]

		for field in bool_fields:
			dataset[field] = dataset[field].apply(
				lambda x:
					True if x == 't' or x == 'Y' or x == 'YES'
					else (
						False if x == 'f' or x == 'N' or x == 'NO'
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
		dataset['_dataset_item_type'] = 'URGENT CARE FACILITY'

		# Add country.
		dataset['iso_level_1'] = 'USA'

		# Copy to iso level 2
		dataset['iso_level_2'] = dataset['state']

		# Copy to iso level 3
		dataset['iso_level_3'] = dataset['county']

		# Copy to iso level 4
		dataset['iso_level_4'] = dataset['city']

		return dataset  												# ==>

	# Process 'hifld_us_ports_of_entry' dataset
	def process_hifld_us_ports_of_entry(self, dataset: pd.DataFrame):
		'''
		Process t/f fields into booleans, normalise `geometry` field
		into GeoJSON format and add ISO fields.

		:param dataset: Dataset to process
		'''

		# Rename columns
		columns = map(
			lambda x:
				x[2:] if x[:2] == 'x_'
				else x,
			list(dataset.columns)
		)
		rename = dict()
		for name in columns:
			rename['x_{}'.format(name)] = name
		dataset.rename(columns=rename, inplace=True)

		# Set not available
		dataset.replace(
			to_replace='NOT AVAILABLE',
			value=float('NaN'),
			inplace=True
		)

		# Normalise boolean fields.
		bool_fields = [
			'phoneloc'
		]

		for field in bool_fields:
			dataset[field] = dataset[field].apply(
				lambda x:
					True if x == 't' or x == 'Y' or x == 'YES'
					else (
						False if x == 'f' or x == 'N' or x == 'NO'
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
		dataset['_dataset_item_type'] = 'PORT OF ENTRY'

		# Add country.
		dataset['iso_level_1'] = 'USA'

		# Copy to iso level 2
		dataset['iso_level_2'] = dataset['state']

		# Copy to iso level 3
		dataset['iso_level_3'] = dataset['county']

		# Copy to iso level 4
		dataset['iso_level_4'] = dataset['city']

		return dataset  												# ==>

	# Process 'ihme_hospitalisation' dataset
	def process_ihme_hospitalisation(self, dataset: pd.DataFrame):
		'''
		Load as-is.

		:param dataset: Dataset to process
		'''

		# Add country.
		dataset['iso_level_1'] = 'USA'

		# Add province
		dataset['iso_level_2'] = \
			dataset['location'].apply(
				lambda x:
					'USA-{}'.format(self.us_states[x]) if x in self.us_states.keys()
					else None
			)

		# Drop location name
		dataset.drop(columns='location_name', inplace=True)

		return dataset  												# ==>

	# Process 'nextstrain_phylogeny' dataset
	def process_nextstrain_phylogeny(self, dataset: pd.DataFrame):
		'''
		Load as-is.

		:param dataset: Dataset to process
		'''

		# Normalise non available values.
		dataset.replace(
			to_replace='?',
			value=float('NaN'),
			inplace=True
		)

		# Normalise sex.
		dataset['sex_male'] = dataset['sex'].apply(
			lambda x:
				True if x == 'Male'
				else (
					False if x == 'Female'
					else None
				)
		)

		return dataset  												# ==>

	# Process 'owd_tests_conducted' dataset
	def process_owd_tests_conducted(self, dataset: pd.DataFrame):
		'''
		Load as-is.

		:param dataset: Dataset to process
		'''

		# Add country.
		dataset['iso_level_1'] = dataset['code']

		return dataset  												# ==>

	# Process 'wfp_travel_restrictions' dataset
	def process_wfp_travel_restrictions(self, dataset: pd.DataFrame):
		'''
		Load as-is.

		:param dataset: Dataset to process
		'''

		# Add country.
		dataset['iso_level_1'] = dataset['iso3']

		return dataset  												# ==>

	# Process 'cdc_cities_census_tract_level' dataset
	def process_cdc_cities_census_tract_level(self, dataset: pd.DataFrame):
		'''
		Load as-is.

		:param dataset: Dataset to process
		'''

		# Get coordinates
		dataset['latitude'] = \
			dataset['geolocation'].apply(
				lambda x:
					None if pd.isna(x)
					else float(
						re.search('\(([-]?\d+\.\d+),[ ]?([-]?\d+\.\d+)\)', x)\
							.group(1)
					)
			)
		dataset['longitude'] = \
			dataset['geolocation'].apply(
				lambda x:
					None if pd.isna(x)
					else float(
						re.search('\(([-]?\d+\.\d+),[ ]?([-]?\d+\.\d+)\)', x)\
							.group(2)
					)
			)

		# Add country.
		dataset['iso_level_1'] = 'USA'

		# Add state.
		dataset['iso_level_2'] = dataset['stateabbr']

		# Add city
		dataset['iso_level_3'] = dataset['placename']

		return dataset  												# ==>

	# Process 'cdc_global_adult_tobacco_survey' dataset
	def process_cdc_global_adult_tobacco_survey(self, dataset: pd.DataFrame):
		'''
		Load as-is.

		:param dataset: Dataset to process
		'''

		# Get coordinates
		dataset['latitude'] = \
			dataset['geolocation'].apply(
				lambda x:
					None if pd.isna(x)
					else float(
						re.search('\(([-]?\d+\.\d+),[ ]?([-]?\d+\.\d+)\)', x)\
							.group(1)
					)
			)
		dataset['longitude'] = \
			dataset['geolocation'].apply(
				lambda x:
					None if pd.isna(x)
					else float(
						re.search('\(([-]?\d+\.\d+),[ ]?([-]?\d+\.\d+)\)', x)\
							.group(2)
					)
			)

		# Add country.
		dataset['iso_level_1'] = \
			dataset['countryabbr'].apply(
				lambda x:
					None if x is None
					else x.upper()
			)

		return dataset  												# ==>

	# Process 'cdc_global_youth_tobacco_survey' dataset
	def process_cdc_global_youth_tobacco_survey(self, dataset: pd.DataFrame):
		'''
		Load as-is.

		:param dataset: Dataset to process
		'''

		# Get coordinates
		dataset['latitude'] = \
			dataset['geolocation'].apply(
				lambda x:
					None if pd.isna(x)
					else float(
						re.search('\(([-]?\d+\.\d+),[ ]?([-]?\d+\.\d+)\)', x)\
							.group(1)
					)
			)
		dataset['longitude'] = \
			dataset['geolocation'].apply(
				lambda x:
					None if pd.isna(x)
					else float(
						re.search('\(([-]?\d+\.\d+),[ ]?([-]?\d+\.\d+)\)', x)\
							.group(2)
					)
			)

		# Add country.
		dataset['iso_level_1'] = \
			dataset['countryabbr'].apply(
				lambda x:
					None if x is None
					else x.upper()
			)

		return dataset  												# ==>

	# Process 'cdc_behavioral_risk_factor_surveillance' dataset
	def process_cdc_behavioral_risk_factor_surveillance(self, dataset: pd.DataFrame):
		'''
		Load as-is.

		:param dataset: Dataset to process
		'''

		# Get coordinates
		dataset['latitude'] = \
			dataset['geolocation'].apply(
				lambda x:
					None if pd.isna(x)
					else float(
						re.search('\(([-]?\d+\.\d+),[ ]?([-]?\d+\.\d+)\)', x)\
							.group(1)
					)
			)
		dataset['longitude'] = \
			dataset['geolocation'].apply(
				lambda x:
					None if pd.isna(x)
					else float(
						re.search('\(([-]?\d+\.\d+),[ ]?([-]?\d+\.\d+)\)', x)\
							.group(2)
					)
			)

		# Add country.
		dataset['iso_level_1'] = 'USA'

		# Add state.
		dataset['iso_level_2'] = \
			dataset['locationabbr'].apply(
				lambda x:
					None if x == 'US'
					else x
			)

		return dataset  												# ==>

	# Process 'cdc_chronic_disease_indicators' dataset
	def process_cdc_chronic_disease_indicators(self, dataset: pd.DataFrame):
		'''
		Load as-is.

		:param dataset: Dataset to process
		'''

		# Get coordinates
		dataset['latitude'] = \
			dataset['geolocation'].apply(
				lambda x:
					None if pd.isna(x)
					else float(
						re.search('\(([-]?\d+\.\d+),[ ]?([-]?\d+\.\d+)\)', x)\
							.group(1)
					)
			)
		dataset['longitude'] = \
			dataset['geolocation'].apply(
				lambda x:
					None if pd.isna(x)
					else float(
						re.search('\(([-]?\d+\.\d+),[ ]?([-]?\d+\.\d+)\)', x)\
							.group(2)
					)
			)

		# Add country.
		dataset['iso_level_1'] = 'USA'

		# Add state.
		dataset['iso_level_2'] = \
			dataset['locationabbr'].apply(
				lambda x:
					None if x == 'US'
					else x
			)

		return dataset  												# ==>

	# Process 'usafacts_covid_19_by_county' dataset
	def process_usafacts_covid_19_by_county(self, dataset: pd.DataFrame):
		'''
		Load as-is.

		:param dataset: Dataset to process
		'''

		# Normalise geometry field.
		dataset['geometry'] = dataset['geometry'].apply(
			lambda x:
				None if pd.isna(x)
				else self.parse_geometries(x)
		)

		# Add country.
		dataset['iso_level_1'] = 'USA'

		# Add state.
		dataset['iso_level_2'] = dataset['state_name']

		# Add county.
		dataset['iso_level_3'] = dataset['county_name']

		return dataset  												# ==>

	# Process 'who_tuberculosis_case_notifications' dataset
	def process_who_tuberculosis_case_notifications(self, dataset: pd.DataFrame):
		'''
		Load as-is.

		:param dataset: Dataset to process
		'''

		# Add country.
		dataset['iso_level_1'] = dataset['iso3']

		return dataset  												# ==>

	# Process 'wb_global_population' dataset
	def process_wb_global_population(self, dataset: pd.DataFrame):
		'''
		Load as-is.

		:param dataset: Dataset to process
		'''

		# Reshape the dataset
		dataset = dataset \
			.melt(id_vars=['country', 'country_code']) \
			.rename(columns=dict(variable='year')) \
			.dropna(axis=0, how='any')

		# Fix year
		dataset['year'] = dataset['year'].apply(lambda x: int(x[5:]))

		# Add country.
		dataset['iso_level_1'] = dataset['country_code']

		return dataset													# ==>

	# Process 'wb_health_population_summary' dataset
	def process_wb_health_population_summary(self, dataset: pd.DataFrame):
		'''
		Load as-is.

		:param dataset: Dataset to process
		'''

		# Add country.
		dataset['iso_level_1'] = dataset['country_code']

		return dataset													# ==>

	# Process 'wb_world_development_indicators_summary' dataset
	def process_wb_world_development_indicators_summary(self, dataset: pd.DataFrame):
		'''
		Load as-is.

		:param dataset: Dataset to process
		'''

		# Add country.
		dataset['iso_level_1'] = dataset['country_code']

		return dataset													# ==>

	# Process 'border_wait_times_at_us_canada_border' dataset
	def process_border_wait_times_at_us_canada_border(self, file: str) -> int:
		'''
		Normalise `geometry` fields.

		:param file: Dataset file path
		'''

		# Init globals
		record_count = 0
		col_name_border = 'border_ports'
		col_name_wait_times = 'border_times'

		# Get/create collection
		if self.db.has_collection(col_name_border):
			collection = self.db.collection(col_name_border)
		else:
			collection = self.db.create_collection(col_name_border, edge=False)

		# Truncate collection
		collection.truncate()

		# Read dataset
		df = pd.read_csv(file)

		# Border columns
		can_columns = [
			'borderid', 'canadaport', 'canadaborderzone',
			'can_iso_3166_2', 'borderlatitude', 'borderlongitude',
			'bordergeohash', 'version'
		]
		usa_columns = [
			'borderid', 'americaport', 'americaborderzone',
			'us_iso_3166_2', 'borderlatitude', 'borderlongitude',
			'bordergeohash', 'version'
		]

		# Init border dataset
		borders = pd.DataFrame()

		# Create borders dataset
		for borderid in df['borderid'].unique():
			group = df[df['borderid'] == borderid].iloc[0][can_columns].copy()
			group['port'] = group['canadaport']
			group['geometry'] = group['canadaborderzone']
			group['border_ref'] = 'BORDER-CAN-{}'.format(borderid)
			group['_key'] = group['border_ref']
			group['iso_level_1'] = 'CAN'
			group['iso_level_2'] = group['can_iso_3166_2']
			borders = borders.append(group, ignore_index=True)

			group = df[df['borderid'] == borderid].iloc[0][usa_columns].copy()
			group['port'] = group['americaport']
			group['geometry'] = group['americaborderzone']
			group['border_ref'] = 'BORDER-USA-{}'.format(borderid)
			group['_key'] = group['border_ref']
			group['iso_level_1'] = 'USA'
			group['iso_level_2'] = group['us_iso_3166_2']
			borders = borders.append(group, ignore_index=True)

		# Remove specific geometry columns
		borders.drop(
			columns=[
				'canadaport', 'americaport',
				'can_iso_3166_2', 'us_iso_3166_2',
				'canadaborderzone', 'americaborderzone'
			],
			inplace=True)

		# Normalise geometry field.
		borders['geometry'] = borders['geometry'].apply(
			lambda x: self.parse_geometries(x)
		)

		# Indicate has shape
		borders['_dataset_has_shape'] = True
		borders['_dataset_item_type'] = 'PORT OF ENTRY'

		# Save border columns
		columns = borders.columns
		if self.ix.has(col_name_border):
			self.ix.update(dict(
				_key=col_name_border,
				file=file,
				collection=col_name_border,
				columns=list(list(columns)),
				has_shape=True
			))

		# Has no index
		else:
			self.ix.insert(dict(
				_key=col_name_border,
				file=file,
				collection=col_name_border,
				columns=list(list(columns)),
				has_shape=True
			))

		# Load the border data
		collection.import_bulk(
			[
				{k: v for k, v in m.items() if pd.notnull(v)}
				for m in borders.to_dict(orient='rows')
			],
			on_duplicate='error',
			sync=True
		)
		record_count += collection.count()

		# Get/create collection
		if self.db.has_collection(col_name_wait_times):
			collection = self.db.collection(col_name_wait_times)
		else:
			collection = self.db.create_collection(col_name_wait_times, edge=True)

		# Truncate collection
		collection.truncate()

		# Columns to drop
		drop_columns = [
			'canadaport', 'americaport', 'tripdirection',
			'canadaborderzone', 'can_iso_3166_2',
			'americaborderzone', 'us_iso_3166_2',
			'borderlatitude', 'borderlongitude',
			'bordergeohash', 'version'
		]

		# Handle canada waiting times
		group = group = df[df['tripdirection'] == 'Canada to US'].copy()
		group['_from'] = \
			group['borderid'].apply(
				lambda x: '{}/BORDER-CAN-{}'.format(col_name_border, x)
			)
		group['_to'] = \
			group['borderid'].apply(
				lambda x: '{}/BORDER-USA-{}'.format(col_name_border, x)
			)
		group.drop(
			columns=drop_columns,
			inplace=True
		)

		# Load the border data
		collection.import_bulk(
			[
				{k: v for k, v in m.items() if pd.notnull(v)}
				for m in group.to_dict(orient='rows')
			],
			on_duplicate='error',
			sync=True
		)
		record_count += collection.count()

		# Handle usa waiting times
		group = group = df[df['tripdirection'] == 'US to Canada'].copy()
		group['_from'] = \
			group['borderid'].apply(
				lambda x: '{}/USA-{}'.format(col_name_border, x)
			)
		group['_to'] = \
			group['borderid'].apply(
				lambda x: '{}/CAN-{}'.format(col_name_border, x)
			)
		group.drop(
			columns=drop_columns,
			inplace=True
		)

		# Save border columns
		columns = group.columns
		if self.ix.has(col_name_wait_times):
			self.ix.update(dict(
				_key=col_name_wait_times,
				file=file,
				collection=col_name_wait_times,
				columns=list(list(columns)),
				has_shape=False
			))

		# Has no index
		else:
			self.ix.insert(dict(
				_key=col_name_wait_times,
				file=file,
				collection=col_name_wait_times,
				columns=list(list(columns)),
				has_shape=False
			))

		# Load the border data
		collection.import_bulk(
			[
				{k: v for k, v in m.items() if pd.notnull(v)}
				for m in group.to_dict(orient='rows')
			],
			on_duplicate='error',
			sync=True
		)
		record_count += collection.count()

		return record_count												# ==>
