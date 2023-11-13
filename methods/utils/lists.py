import pandas as pd

model_names = ['obs_pub_nhmv10_ObsScaled', 'obs_pub_nwmv21_ObsScaled',
               'obs_pub_nhmv10_ObsScaled_ensemble', 'obs_pub_nwmv21_ObsScaled_ensemble']


nyc_reservoirs = ['cannonsville', 'pepacton', 'neversink']
upper_basin_reservoirs = ['prompton', 'mongaupeCombined', 'wallenpaupack', 
                          'shoholaMarsh'] 
mid_basin_reservoirs = ['fewalter', 'hopatcong', 'beltzvilleCombined',
                        'merrillCreek', 'nockamixon', 'assunpink']
lower_basin_reservoirs = ['blueMarsh', 'ontelaunee', 'stillCreek', 
                          'greenLane']

model_datasets = ['nhmv10', 'nwmv21', 'pub_nhmv10', 'pub_nwmv21', 'pub_nhmv10_ens', 'pub_nwmv21_ens']
single_datasets = ['nhmv10', 'nwmv21', 'pub_nhmv10', 'pub_nwmv21']
pub_datasets = ['pub_nhmv10', 'pub_nwmv21', 'pub_nhmv10', 'pub_nwmv21']
ensemble_datasets = ['pub_nhmv10_ens', 'pub_nwmv21_ens']

# Historic DRBC drought events
event_types = [['Emergency', '1965-07-07', '1967-03-15'],
    ['Watch', '1980-10-17', '1982-04-27'], 
    ['Emergency', '1981-01-15', '1982-04-27'],
    ['Watch', '1982-11-13', '1983-03-27'], 
    ['Watch', '1983-11-09', '1983-12-20'],
    ['Watch', '1985-01-23', '1985-12-18'], 
    ['Warning', '1985-02-07', '1985-12-18'], 
    ['Emergency', '1985-05-13', '1985-12-18'],
    ['Watch', '1989-01-16', '1989-05-12'],
    ['Warning', '1989-02-05', '1989-05-12'],
    ['Watch', '1991-09-13', '1992-06-17'], 
    ['Warning', '1991-11-07', '1992-06-17'],
    ['Watch', '1993-09-21', '1993-12-06'],
    ['Watch', '1995-09-15', '1995-11-12'], 
    ['Warning', '1995-10-13', '1995-11-12'],
    ['Watch', '1997-10-27', '1998-01-13'],
    ['Watch', '1998-12-14', '1999-02-02'],
    ['Warning', '1998-12-23', '1999-02-02'],
    ['Emergency', '1999-08-18', '1999-09-30'],
    ['Watch', '2001-10-29', '2002-11-25'], 
    ['Warning', '2001-11-04', '2002-11-25'], 
    ['Emergency', '2001-12-18', '2002-11-25'],
    ['Watch', '2016-11-23', '2017-01-18']]

drbc_droughts = pd.DataFrame(event_types, columns=['event_type', 'start_date', 'end_date'])
drbc_droughts.head()