from __future__ import annotations
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from .common import read_yaml, resolve_path, write_json, sha256

METEO_CANDIDATES=['temperature','humidity','wind_speed','meridian_wind_speed','latitudinal_wind_speed',
                  'precipitation','ground_pressure','dew_point_temperature','soil_temp','soil_moisture']
SOLAR_OBSERVED=['light','net_solar_radiation','theoretical_radiation']

def solar_features(ts, lat, lon, tz_hours):
    """NOAA-style approximate solar geometry; output uses cyclic encodings."""
    ts=pd.DatetimeIndex(ts)
    day=ts.dayofyear.to_numpy(float)
    hour=ts.hour.to_numpy(float)+ts.minute.to_numpy(float)/60+ts.second.to_numpy(float)/3600
    b=np.deg2rad(360*(day-81)/364.0)
    eot=9.87*np.sin(2*b)-7.53*np.cos(b)-1.5*np.sin(b)
    local_solar=hour+(4*(lon-15*tz_hours)+eot)/60
    decl=np.deg2rad(23.45*np.sin(np.deg2rad(360*(284+day)/365.0)))
    ha=np.deg2rad(15*(local_solar-12))
    latr=np.deg2rad(lat)
    sin_alt=np.sin(latr)*np.sin(decl)+np.cos(latr)*np.cos(decl)*np.cos(ha)
    alt=np.arcsin(np.clip(sin_alt,-1,1))
    az=np.arctan2(np.sin(ha), np.cos(ha)*np.sin(latr)-np.tan(decl)*np.cos(latr))+np.pi
    vals={'declination':decl,'hour_angle':ha,'altitude':alt,'azimuth':az}
    out={}
    for k,v in vals.items(): out[f'solar_{k}_sin']=np.sin(v); out[f'solar_{k}_cos']=np.cos(v)
    # Include daylight indicator; avoids interpreting below-horizon azimuth as forcing.
    out['solar_daylight']=(np.rad2deg(alt)>0).astype(float)
    return pd.DataFrame(out,index=ts)

def calendar_features(ts):
    ts=pd.DatetimeIndex(ts); doy=ts.dayofyear.to_numpy(float); hour=(ts.hour+ts.minute/60).to_numpy(float)
    return pd.DataFrame({'hour_sin':np.sin(2*np.pi*hour/24),'hour_cos':np.cos(2*np.pi*hour/24),
                         'doy_sin':np.sin(2*np.pi*doy/365.2425),'doy_cos':np.cos(2*np.pi*doy/365.2425)},index=ts)

def sensor_columns(mapping):
    ordered=[]
    for site, spec in mapping['sites'].items():
        for hidx,sid in enumerate(spec['sensor_ids']):
            row={'site':site,'sensor_id':sid,'height_m':float(mapping['height_m'][hidx]),'xy':spec['xy']}
            row['columns']={name:mapping['variables'][name].format(sensor_id=sid)
                            for name in ('temperature','humidity','light')}
            ordered.append(row)
    return ordered

def load_and_merge(data_cfg_path, spatial_cfg_path, allow_unverified=False):
    dc=read_yaml(data_cfg_path); mp=read_yaml(spatial_cfg_path)
    if not mp.get('verified',False) and not allow_unverified:
        raise RuntimeError('Spatial coordinates/mapping are marked unverified. Review the figure-derived mapping, then explicitly pass --allow-unverified-layout for exploratory data construction.')
    base=Path(data_cfg_path).resolve().parent
    indoor_path=resolve_path(dc['indoor_csv'],base); outdoor_path=resolve_path(dc['outdoor_csv'],base)
    indoor=pd.read_csv(indoor_path); outdoor=pd.read_csv(outdoor_path)
    tcol=dc.get('timestamp_column','timestamp')
    # pandas handles the two source timestamp formats with format='mixed' on modern releases;
    # coerce row-wise fallback keeps compatibility with older pandas.
    def parse_time(s):
        # Outdoor source uses DD/MM/YYYY while indoor uses ISO YYYY-MM-DD.
        # dayfirst=True resolves the ambiguous dates correctly and does not alter ISO timestamps.
        try: return pd.to_datetime(s,format='mixed',errors='coerce',dayfirst=True)
        except (TypeError,ValueError): return pd.to_datetime(s,errors='coerce',dayfirst=True)
    indoor[tcol]=parse_time(indoor[tcol]); outdoor[tcol]=parse_time(outdoor[tcol])
    indoor=indoor.dropna(subset=[tcol]).sort_values(tcol).drop_duplicates(tcol,keep='last')
    outdoor=outdoor.dropna(subset=[tcol]).sort_values(tcol).drop_duplicates(tcol,keep='last')
    merged=indoor.merge(outdoor,on=tcol,how='inner',suffixes=('','_outdoor'),validate='one_to_one').sort_values(tcol).reset_index(drop=True)
    nodes=sensor_columns(mp)
    missing=[col for n in nodes for col in n['columns'].values() if col not in merged.columns]
    if missing: raise ValueError(f'Missing mapped sensor columns: {missing}')
    ycols=[n['columns'][k] for n in nodes for k in ('temperature','humidity','light')]
    # Preserve feature order as node-major [temperature, humidity, light].
    meteo=[c for c in METEO_CANDIDATES if c in merged.columns]
    solar=solar_features(merged[tcol],dc['latitude_deg'],dc['longitude_deg'],dc['timezone_offset_hours'])
    cal=calendar_features(merged[tcol])
    for f in (solar,cal): merged[f.columns]=f.to_numpy()
    meta={'indoor_path':str(indoor_path),'outdoor_path':str(outdoor_path),'indoor_sha256':sha256(indoor_path),
          'outdoor_sha256':sha256(outdoor_path),'indoor_rows':len(indoor),'outdoor_rows':len(outdoor),
          'merged_rows':len(merged),'start':str(merged[tcol].min()),'end':str(merged[tcol].max()),
          'target_columns':ycols,'nodes':nodes,'meteorology_columns':meteo,
          'missing_by_target':merged[ycols].isna().sum().to_dict(),'mapping_verified':mp.get('verified',False),
          'spatial_coordinate_units':mp.get('coordinate_units'),'timestamp_column':tcol}
    return merged,meta,dc,mp

def audit_data(data_cfg_path, spatial_cfg_path, output_path=None):
    df,meta,dc,mp=load_and_merge(data_cfg_path,spatial_cfg_path,allow_unverified=True)
    ts=pd.to_datetime(df[meta['timestamp_column']]); dif=ts.diff().dropna().dt.total_seconds()/60
    meta['cadence_minutes']={'median':float(dif.median()) if len(dif) else None,
                             'common':dif.value_counts().head(10).to_dict(),
                             'expected':dc.get('expected_cadence_minutes')}
    meta['target_missing_total']=int(df[meta['target_columns']].isna().sum().sum())
    meta['target_nonfinite_total']=int((~np.isfinite(df[meta['target_columns']].apply(pd.to_numeric,errors='coerce'))).sum().sum())
    meta['outdoor_columns']=list(pd.read_csv(meta['outdoor_path'],nrows=0).columns)
    meta['solar_feature_columns']=[c for c in df.columns if c.startswith('solar_')]
    meta['calendar_feature_columns']=[c for c in df.columns if c in ('hour_sin','hour_cos','doy_sin','doy_cos')]
    if output_path: write_json(meta,output_path)
    return meta

def make_feature_frames(df,meta,feature_mode='main_solar',include_outdoor_solar=True):
    ycols=meta['target_columns']; meteo=meta['meteorology_columns']
    met=meteo.copy()
    if include_outdoor_solar:
        met += [c for c in SOLAR_OBSERVED if c in df.columns]
    # For the calendar-versus-geometry contrast, no measured or theoretical solar signal
    # is exposed in any arm. This controls the feature set outside the factor of interest.
    if feature_mode in ('no_temporal','calendar','solar_geometry','calendar_plus_solar'):
        met=[c for c in met if c not in SOLAR_OBSERVED]
    feats={'indoor':ycols,'meteo':met}
    if feature_mode in ('calendar','calendar_plus_solar'):
        feats['calendar']=[c for c in ('hour_sin','hour_cos','doy_sin','doy_cos') if c in df]
    if feature_mode in ('solar_geometry','calendar_plus_solar','main_solar','geometry_plus_raw','full','full_ceemdan','full_ceemdan_geometry'):
        feats['solar']=[c for c in df if c.startswith('solar_')]
    cols=sum(feats.values(),[])
    x=df[cols].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=np.float32)
    y=df[ycols].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=np.float32).reshape(len(df),len(meta['nodes']),3)
    # node feature dimensions: indoor's own three measurements; global channels are broadcast
    # by the dataset loader, not duplicated in the serialized source tensor.
    blocks={'indoor_indices':list(range(len(ycols))),'global_indices':list(range(len(ycols),len(cols))),
            'global_columns':cols[len(ycols):],'feature_columns':cols}
    global_x=x[:,len(ycols):]
    return y,global_x,blocks

def fit_train_scalers(df, meta, train_end, global_cols):
    ycols=meta['target_columns']; y=df[ycols].apply(pd.to_numeric,errors='coerce').to_numpy(np.float64)
    g=df[global_cols].apply(pd.to_numeric,errors='coerce').to_numpy(np.float64) if global_cols else np.zeros((len(df),0))
    ym=np.nanmean(y[:train_end],axis=0); ys=np.nanstd(y[:train_end],axis=0)
    gm=np.nanmean(g[:train_end],axis=0) if g.shape[1] else np.zeros(0); gs=np.nanstd(g[:train_end],axis=0) if g.shape[1] else np.ones(0)
    ys=np.where(ys<1e-8,1,ys); gs=np.where(gs<1e-8,1,gs)
    return {'y_mean':ym.astype(np.float32),'y_std':ys.astype(np.float32),'g_mean':gm.astype(np.float32),'g_std':gs.astype(np.float32)}

def build_windows(df,meta,feature_mode,history,horizon,split_cfg,include_outdoor_solar=True):
    n=len(df); tr=float(split_cfg['train']); vr=float(split_cfg['validation']); te=float(split_cfg['test'])
    if abs(tr+vr+te-1)>1e-8: raise ValueError('Split ratios must sum to 1')
    train_end=int(n*tr); val_end=int(n*(tr+vr))
    y, gx, blocks=make_feature_frames(df,meta,feature_mode,include_outdoor_solar)
    # Historical covariates are filled only from the past (short forward fill).
    # Labels are never imputed; unresolved NaNs cause affected windows to be skipped.
    gcols=blocks['global_columns']; scaler=fit_train_scalers(df,meta,train_end,gcols)
    ydf=pd.DataFrame(y.reshape(n,-1)); gdf=pd.DataFrame(gx)
    bounds={'train':(0,train_end),'validation':(train_end,val_end),'test':(val_end,n)}
    # Historical inputs may be carried forward causally. Labels are never imputed.
    y_hist=ydf.ffill(limit=2).to_numpy(np.float32).reshape(n,len(meta['nodes']),3)
    g_hist=gdf.ffill(limit=2).to_numpy(np.float32) if gx.shape[1] else gx.copy()
    arrays={}; maxgap=int(2); skipped_history={}; skipped_target={}
    for split,(lo,hi) in bounds.items():
        local_y=y[lo:hi]
        # Context can come from before the split boundary; predictions/labels cannot cross it.
        target_first=max(lo,history); target_last=hi-horizon
        starts=np.arange(target_first-history,target_last-history+1,dtype=np.int64)
        if not len(starts): raise ValueError(f'No windows in split {split}; n={n}, H={history}, P={horizon}')
        X=[]; Y=[]; target_starts=[]; skipped_history[split]=0; skipped_target[split]=0
        for s in starts:
            e=s+history; ys0=e; ye=ys0+horizon
            # The forecast targets must belong to this split; context may precede it.
            if ys0<lo or ye>hi: continue
            # Context crossing a split uses only current/past observations (forward fill).
            hist_y_missing=(~np.isfinite(y[s:e])).astype(np.float32)
            hist_g_missing=(~np.isfinite(gx[s:e])).astype(np.float32) if gx.shape[1] else np.zeros((history,0),np.float32)
            hist_y=y_hist[s:e].copy(); hist_g=g_hist[s:e].copy()
            if not np.isfinite(hist_y).all() or not np.isfinite(hist_g).all():
                skipped_history[split]+=1
                continue
            hist_y=(hist_y-scaler['y_mean'].reshape(1,len(meta['nodes']),3))/scaler['y_std'].reshape(1,len(meta['nodes']),3)
            hist_g=(hist_g-scaler['g_mean'])/scaler['g_std'] if gx.shape[1] else hist_g
            target=local_y[ys0-lo:ye-lo].copy()
            if not np.isfinite(target).all():
                skipped_target[split]+=1
                continue
            target=(target-scaler['y_mean'].reshape(1,len(meta['nodes']),3))/scaler['y_std'].reshape(1,len(meta['nodes']),3)
            hist_x=np.concatenate([hist_y,hist_g[:,None,:].repeat(len(meta['nodes']),axis=1),
                                   hist_y_missing,hist_g_missing[:,None,:].repeat(len(meta['nodes']),axis=1)],axis=-1)
            if feature_mode in ('ceemdan_causal','full_ceemdan','full_ceemdan_geometry'):
                # Causal per-window decomposition of one measured outdoor irradiance proxy;
                # the same three forcing components are broadcast to all indoor sites.
                # This avoids hundreds of thousands of per-sensor CEEMDAN calls.
                try:
                    from PyEMD import CEEMDAN
                except ImportError as e:
                    raise RuntimeError('CEEMDAN requested but EMD-signal/PyEMD is not installed; no surrogate fallback is allowed.') from e
                source_name=next((c for c in ('light','net_solar_radiation') if c in gcols),None)
                if source_name is None: raise ValueError('CEEMDAN mode needs outdoor light or net_solar_radiation in the selected source data.')
                source_idx=gcols.index(source_name)
                # The decomposition context may be longer than the model history so
                # short H=6 forecasting windows still have a valid causal CEEMDAN input.
                context_len=max(12,history); context_start=max(0,e-context_len)
                context_raw=g_hist[context_start:e,source_idx].astype(np.float64)
                if len(context_raw)<8 or not np.isfinite(context_raw).all() or np.std(context_raw)<1e-9:
                    skipped_history[split]+=1
                    continue
                sig=(context_raw-scaler['g_mean'][source_idx])/scaler['g_std'][source_idx]
                digest=hashlib.sha256(sig.tobytes()+b'ceemdan-v1-trials10-maximf3').hexdigest()
                cache_dir=Path(__file__).resolve().parents[1]/'data'/'cache'/'ceemdan'; cache_dir.mkdir(parents=True,exist_ok=True)
                cache_file=cache_dir/f'{digest}.npz'
                if cache_file.exists(): common_imfs=np.load(cache_file)['components'].astype(np.float32)
                else:
                    ce=CEEMDAN(trials=10,parallel=False); ce.noise_seed(int(digest[:8],16) % (2**31-1))
                    imfs=ce.ceemdan(sig,max_imf=3)
                    if imfs.ndim==1: imfs=imfs[None,:]
                    if imfs.shape[0]<3: imfs=np.concatenate([imfs,np.zeros((3-imfs.shape[0],len(sig)))],axis=0)
                    common_imfs=imfs[:3].T.astype(np.float32); np.savez_compressed(cache_file,components=common_imfs)
                common_imfs=common_imfs[-history:]
                if len(common_imfs)<history:
                    common_imfs=np.concatenate([np.repeat(common_imfs[:1],history-len(common_imfs),axis=0),common_imfs],axis=0)
                hist_x=np.concatenate([hist_x,common_imfs[:,None,:].repeat(len(meta['nodes']),axis=1)],axis=-1)
            X.append(hist_x)
            Y.append(target); target_starts.append(ys0)
        arrays[split]={'X':np.asarray(X,np.float32),'Y':np.asarray(Y,np.float32),'target_start':np.asarray(target_starts,np.int64)}
    # explicit spatial metadata carried into model and prediction outputs
    heights=np.asarray([n0['height_m'] for n0 in meta['nodes']],np.float32)
    z=(heights-heights.min())/max(float(heights.max()-heights.min()),1e-8)
    coords=np.asarray([n0['xy']+[float(z[i])] for i,n0 in enumerate(meta['nodes'])],np.float32)
    return arrays,scaler,{'history':history,'horizon':horizon,'feature_mode':feature_mode,'feature_columns':blocks['feature_columns'],
                         'global_columns':gcols,'nodes':meta['nodes'],'coords':coords.tolist(),'mapping_verified':meta['mapping_verified'],
                         'split_bounds':bounds,'n_rows':n,'split_ratios':split_cfg,
                         'skipped_windows_history_missing':skipped_history,'skipped_windows_target_missing':skipped_target,
                         'max_causal_forward_fill_steps':maxgap,'input_missingness_masks':True}
