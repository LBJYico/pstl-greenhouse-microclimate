import numpy as np
import pandas as pd
from pstl_experiments.data import build_windows, sensor_columns

def test_sensor_column_mapping_has_27_nodes_and_81_targets():
    mapping={'height_m':[1,2,4], 'variables':{'temperature':'T{sensor_id}','humidity':'H{sensor_id}','light':'L{sensor_id}'},
             'sites':{'A1':{'sensor_ids':['01','02','03'],'xy':[0.2,0.1]},
                      'A2':{'sensor_ids':['04','05','06'],'xy':[0.8,0.1]},
                      'A3':{'sensor_ids':['07','08','09'],'xy':[0.5,0.5]},
                      'A4':{'sensor_ids':['10','11','12'],'xy':[0.2,0.8]},
                      'A5':{'sensor_ids':['13','14','15'],'xy':[0.8,0.8]},
                      'A6':{'sensor_ids':['16','17','18'],'xy':[0.3,0.4]},
                      'A7':{'sensor_ids':['19','20','21'],'xy':[0.7,0.4]},
                      'A8':{'sensor_ids':['22','23','24'],'xy':[0.3,0.6]},
                      'A9':{'sensor_ids':['25','26','27'],'xy':[0.7,0.6]}}}
    nodes=sensor_columns(mapping)
    assert len(nodes)==27
    assert len({n['columns'][k] for n in nodes for k in ('temperature','humidity','light')})==81
    assert [n['height_m'] for n in nodes[:3]]==[1.,2.,4.]

def test_target_windows_stay_inside_chronological_split_and_train_scaler():
    n=60; sensors=[]
    for i in range(3):
        sensors.append({'site':f'A{i+1}','sensor_id':str(i),'height_m':[1.,2.,4.][i], 'xy':[i/2,.5],
                        'columns':{'temperature':f'T{i}','humidity':f'H{i}','light':f'L{i}'}})
    cols=[s['columns'][k] for s in sensors for k in ('temperature','humidity','light')]
    frame=pd.DataFrame({'timestamp':pd.date_range('2022-01-01',periods=n,freq='10min')})
    for j,c in enumerate(cols): frame[c]=np.arange(n,dtype=float)+j
    meta={'target_columns':cols,'nodes':sensors,'meteorology_columns':[],'mapping_verified':True,'spatial_coordinate_units':'test'}
    arrays,scaler,dsmeta=build_windows(frame,meta,'no_temporal',4,3,{'train':.6,'validation':.2,'test':.2},False)
    assert arrays['train']['X'].shape[1:3]==(4,3)
    assert arrays['train']['Y'].shape[1:]==(3,3,3)
    assert arrays['validation']['target_start'].min()>=36
    assert arrays['validation']['target_start'].max()+3<=48
    assert arrays['test']['target_start'].min()>=48
    # Scaler is fitted on the training timeline only, not on held-out target values.
    train_values=frame.iloc[:36][cols].to_numpy()
    np.testing.assert_allclose(scaler['y_mean'],train_values.mean(axis=0),rtol=1e-6)
