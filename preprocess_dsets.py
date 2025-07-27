import os
import numpy as np
import medpy.io as medio
import monai
import torch
import glob
import sys
from data.transforms import PadEqualSizeStackd
from subprocess import call
join = os.path.join


def sup_128(xmin, xmax):
    if xmax - xmin < 128:
        print("#" * 100)
        ecart = int((128 - (xmax - xmin)) / 2)
        xmax = xmax + ecart + 1
        xmin = xmin - ecart
    if xmin < 0:
        xmax -= xmin
        xmin = 0
    return xmin, xmax


def crop(vol):
    if len(vol.shape) == 4:
        vol = np.amax(vol, axis=0)
    assert len(vol.shape) == 3

    x_dim, y_dim, z_dim = tuple(vol.shape)
    x_nonzeros, y_nonzeros, z_nonzeros = np.where(vol != 0)

    x_min, x_max = np.amin(x_nonzeros), np.amax(x_nonzeros)
    y_min, y_max = np.amin(y_nonzeros), np.amax(y_nonzeros)
    z_min, z_max = np.amin(z_nonzeros), np.amax(z_nonzeros)

    x_min, x_max = sup_128(x_min, x_max)
    y_min, y_max = sup_128(y_min, y_max)
    z_min, z_max = sup_128(z_min, z_max)

    return x_min, x_max, y_min, y_max, z_min, z_max


def normalize(vol):
    mask = vol.sum(0) > 0
    for k in range(4):
        x = vol[k, ...]
        y = x[mask]
        x = (x - y.mean()) / y.std()
        vol[k, ...] = x

    return vol


def transformsFuncd(keys, allow_missing_keys):
    keys_seg = []
    
    keys_ = []
    for k in keys:
        if '_seg' in k or k == 'segmentation':
            keys_seg.append(k)
        else:
            keys_.append(k)

    keys = keys_
    print(keys, keys_seg)

    transforms = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys=keys+keys_seg, allow_missing_keys=allow_missing_keys),
        monai.transforms.EnsureChannelFirstd(keys=keys+keys_seg, allow_missing_keys=allow_missing_keys),
        monai.transforms.Orientationd(keys=keys+keys_seg, axcodes="RAS"),
        monai.transforms.Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), allow_missing_keys=allow_missing_keys),
        monai.transforms.Spacingd(keys=keys_seg, pixdim=(1.0, 1.0, 1.0), allow_missing_keys=allow_missing_keys, mode=("nearest")),
        monai.transforms.EnsureTyped(keys=keys_seg, allow_missing_keys=allow_missing_keys, dtype=torch.long),
        PadEqualSizeStackd(keys=keys+keys_seg, allow_missing_keys=allow_missing_keys, cropforeground=True),
        monai.transforms.NormalizeIntensityd(keys=keys, allow_missing_keys=allow_missing_keys),
        # monai.transforms.ScaleIntensityd(keys=['flair', 'dwi', 'adc'], minv=0.0, maxv=1.0),
        # monai.transforms.HistogramNormalized(keys=['flair', 'dwi', 'adc'], allow_missing_keys=True)
        # monai.transforms.Resized(keys=['flair', 'dwi', 'adc'], spatial_size=(128, 128, 128)),
    ])
    return transforms


def preprocess_BRATS(dsetname):
    version = dsetname.lower().replace('brats','')
    src_path = f"/projectnb/vkolagrp/datasets/BraTS{version}"
    tar_path = f"/projectnb/ivc-ml/dlteif/BraTS{version}/BRATS{version}_Training_none_npy"
    folder = f'MICCAI_BraTS{version}_TrainingData'
    HGG_list = os.listdir(join(src_path, 'raw', folder, "HGG"))
    HGG_list = ["HGG/" + x for x in HGG_list]
    LGG_list = os.listdir(join(src_path, 'raw', folder, "LGG"))
    LGG_list = ["LGG/" + x for x in LGG_list]
    name_list = HGG_list + LGG_list
    print(name_list)
    # name_list = os.listdir(join(src_path, 'raw', f'MICCAI_BraTS{version}_TrainingData'))
    # name_list += os.listdir(join(src_path, 'raw', f'MICCAI_BraTS{version}_ValidationData'))
    ext = '.nii.gz' if version == '2020' else '.nii'

    os.makedirs(tar_path, exist_ok=True)

    if not os.path.exists(os.path.join(tar_path, "vol")):
        os.makedirs(os.path.join(tar_path, "vol"))

    if not os.path.exists(os.path.join(tar_path, "seg")):
        os.makedirs(os.path.join(tar_path, "seg"))

    if not os.path.exists(os.path.join(tar_path, "synthseg")):
        os.makedirs(os.path.join(tar_path, "synthseg"))


    transforms = monai.Compose([

    ])

    for file_name in name_list:
        print(file_name)
        if '.csv' in file_name or '.txt' in file_name:
            continue

        if len(glob.glob(join(tar_path, 'vol', file_name + "*.npy"))) > 0:
            print("Exists.. Skipping")
            continue
        if version == '2018':
            if "HGG" in file_name:
                HLG = "HGG_"
            else:
                HLG = "LGG_"
        else:
            HLG = ""
        # if 'Training' in file_name:
        #     folder = f'MICCAI_BraTS{version}_TrainingData'
        # else:    
        #     folder = f'MICCAI_BraTS{version}_ValidationData'
        case_id = file_name.split("/")[-1]
        flair, flair_header = medio.load(
            os.path.join(src_path, 'raw', folder, file_name, case_id + f"_flair{ext}")
        )
        t1ce, t1ce_header = medio.load(
            os.path.join(src_path, 'raw', folder, file_name, case_id + f"_t1ce{ext}")
        )
        t1, t1_header = medio.load(
            os.path.join(src_path, 'raw', folder, file_name, case_id + f"_t1{ext}")
        )
        t2, t2_header = medio.load(
            os.path.join(src_path, 'raw', folder, file_name, case_id + f"_t2{ext}")
        )

        vol = np.stack((flair, t1ce, t1, t2), axis=0).astype(np.float32)
        if 'Training' in folder:
            x_min, x_max, y_min, y_max, z_min, z_max = crop(vol)
            vol = vol[:, x_min:x_max, y_min:y_max, z_min:z_max]
        vol1 = normalize(vol)
        vol1 = vol.transpose(1, 2, 3, 0)
        print(vol1.shape)

        if 'Training' in file_name:
            seg, seg_header = medio.load(
                os.path.join(src_path, 'raw', folder, file_name, case_id + f"_seg{ext}")
            )
            seg = seg.astype(np.uint8)
            seg1 = seg[x_min:x_max, y_min:y_max, z_min:z_max]
            seg1[seg1 == 4] = 3

        
        flairseg, flairseg_header = medio.load(
            os.path.join(src_path, 'segmentations', folder, file_name, case_id + f"_flair_seg{ext}")
        )
        t1ceseg, t1ceseg_header = medio.load(
            os.path.join(src_path, 'segmentations', folder, file_name, case_id + f"_t1ce_seg{ext}")
        )
        t1seg, t1seg_header = medio.load(
            os.path.join(src_path, 'segmentations', folder, file_name, case_id + f"_t1_seg{ext}")
        )
        t2seg, t2seg_header = medio.load(
            os.path.join(src_path, 'segmentations', folder, file_name, case_id + f"_t2_seg{ext}")
        )

        
        volseg = np.stack((flairseg, t1ceseg, t1seg, t2seg), axis=0).astype(np.uint16)
        if 'Training' in folder:
            volseg = volseg[:, x_min:x_max, y_min:y_max, z_min:z_max]
        volseg1 = volseg.transpose(1, 2, 3, 0)
        print(volseg1.shape)

        print(file_name)
        np.save(os.path.join(tar_path, "vol", HLG + case_id + "_vol.npy"), vol1)
        if 'Training' in file_name:
            np.save(os.path.join(tar_path, "seg", HLG + case_id + "_seg.npy"), seg1)
        np.save(os.path.join(tar_path, "synthseg", HLG + case_id + "_vol_synthseg.npy"), volseg1)


def preprocess_GliomaPost(dsetname):
    src_path = f"/projectnb/ivc-ml/dlteif/MU-Glioma-Post/raw"
    tar_path = f"/projectnb/ivc-ml/dlteif/MU-Glioma-Post/MU-Glioma-Post_npy"
    folder = 'MU-Glioma-Post'
    name_list = os.listdir(join(src_path))
    print(name_list)
    # name_list = os.listdir(join(src_path, 'raw', f'MICCAI_BraTS{version}_TrainingData'))
    # name_list += os.listdir(join(src_path, 'raw', f'MICCAI_BraTS{version}_ValidationData'))
    ext = '.nii.gz'

    os.makedirs(tar_path, exist_ok=True)

    if not os.path.exists(os.path.join(tar_path, "vol")):
        os.makedirs(os.path.join(tar_path, "vol"))

    if not os.path.exists(os.path.join(tar_path, "seg")):
        os.makedirs(os.path.join(tar_path, "seg"))

    if not os.path.exists(os.path.join(tar_path, "synthseg")):
        os.makedirs(os.path.join(tar_path, "synthseg"))

    for file_name in name_list:
        print(file_name)
        if '.csv' in file_name or '.txt' in file_name:
            continue
        
        
        case_id = file_name.split("/")[-1]
        for subfolder in os.listdir(join(src_path, case_id))[::-1]:
            timept = subfolder.split("/")[-1]
            print(case_id, timept)

            if not os.path.exists(os.path.join(tar_path, "vol", f"{case_id}_{timept}_vol.npy")):
                
                flair, flair_header = medio.load(
                    os.path.join(src_path, file_name, timept, case_id + f"_{timept}_brain_t2f{ext}")
                )
                t1ce, t1ce_header = medio.load(
                    os.path.join(src_path, file_name, timept, case_id + f"_{timept}_brain_t1c{ext}")
                )
                t1, t1_header = medio.load(
                    os.path.join(src_path, file_name, timept, case_id + f"_{timept}_brain_t1n{ext}")
                )
                t2, t2_header = medio.load(
                    os.path.join(src_path, file_name, timept, case_id + f"_{timept}_brain_t2w{ext}")
                )

                vol = np.stack((flair, t1ce, t1, t2), axis=0).astype(np.float32)
                if 'Training' in folder:
                    x_min, x_max, y_min, y_max, z_min, z_max = crop(vol)
                    vol = vol[:, x_min:x_max, y_min:y_max, z_min:z_max]
                vol1 = normalize(vol)
                vol1 = vol.transpose(1, 2, 3, 0)
                print(vol1.shape)
                np.save(os.path.join(tar_path, "vol", f"{case_id}_{timept}_vol.npy"), vol1)

        
            if not os.path.exists(os.path.join(tar_path, "seg", f"{case_id}_{timept}_seg.npy")):
                try:
                    seg, seg_header = medio.load(
                        os.path.join(src_path, file_name, timept, f"{case_id}_{timept}_tumorMask{ext}")
                    )
                    seg = seg.astype(np.uint8)
                    if 'Training' in folder:
                        seg1 = seg[x_min:x_max, y_min:y_max, z_min:z_max]
                    else:
                        seg1 = seg
                    
                    np.save(os.path.join(tar_path, "seg", f"{case_id}_{timept}_seg.npy"), seg1)
                except:
                    pass

            
            if not os.path.exists(os.path.join(tar_path, "synthseg", f"{case_id}_{timept}_vol_synthseg.npy")):
                print(os.path.join(src_path.replace('/raw','/segmentations'), file_name, timept, f"{case_id}_{timept}_brain_t2f_seg{ext}"))
                try:
                    flairseg, flairseg_header = medio.load(
                        os.path.join(src_path.replace('/raw','/segmentations'), file_name, timept, f"{case_id}_{timept}_brain_t2f_seg{ext}")
                    )
                    t1ceseg, t1ceseg_header = medio.load(
                        os.path.join(src_path.replace('/raw','/segmentations'), file_name, timept, f"{case_id}_{timept}_brain_t1c_seg{ext}")
                    )
                    t1seg, t1seg_header = medio.load(
                        os.path.join(src_path.replace('/raw','/segmentations'), file_name, timept, f"{case_id}_{timept}_brain_t1n_seg{ext}")
                    )
                    t2seg, t2seg_header = medio.load(
                        os.path.join(src_path.replace('/raw','/segmentations'), file_name, timept, f"{case_id}_{timept}_brain_t2w_seg{ext}")
                    )

                    volseg = np.stack((flairseg, t1ceseg, t1seg, t2seg), axis=0).astype(np.uint16)
                    if 'Training' in folder:
                        volseg = volseg[:, x_min:x_max, y_min:y_max, z_min:z_max]
                    volseg1 = volseg.transpose(1, 2, 3, 0)
                    print(volseg1.shape)
                    np.save(os.path.join(tar_path, "synthseg", f"{case_id}_{timept}_vol_synthseg.npy"), volseg1)
                except:
                    pass
                
            

def preprocess_ISLES(dsetname):
    version = dsetname.lower().replace('isles','')
    src_path = f"/projectnb/ivc-ml/dlteif/ISLES-{version}"
    tar_path = f"/projectnb/ivc-ml/dlteif/ISLES-{version}/ISLES{version}_npy_Zscorenorm"

    all_list = os.listdir(join(src_path, f'ISLES-{version}RAW'))
    
    os.makedirs(tar_path, exist_ok=True)

    if not os.path.exists(os.path.join(tar_path, "vol")):
        os.makedirs(os.path.join(tar_path, "vol"))

    if not os.path.exists(os.path.join(tar_path, "seg")):
        os.makedirs(os.path.join(tar_path, "seg"))

    if not os.path.exists(os.path.join(tar_path, "synthseg")):
        os.makedirs(os.path.join(tar_path, "synthseg"))
    
    keys = ["flair", "dwi", "adc"]
    keys_seg = [k+"_seg" for k in keys] + ["segmentation"]
    
    transforms = transformsFuncd(keys+keys_seg, allow_missing_keys=True)
    
    for idx, ptID in enumerate(all_list):
        print(f"Subject {idx}/{len(all_list)}")
        print(ptID)
        if not 'sub-strokecase' in ptID:
            print("Skipping..")
            continue
        if len(glob.glob(join(tar_path, 'vol', ptID + "*.npy"))) > 0:
            print("Exists.. Skipping")
            continue
        print(join(src_path, f'ISLES-{version}RAW', ptID + "/*/**/*.nii.gz"))
        ptSessions = os.listdir(join(src_path, f'ISLES-{version}RAW', ptID))
        # print(ptSessions)

        for ptSess in ptSessions:
            visit = {
                'flair': join(src_path, f'ISLES-{version}RAW', ptID, ptSess, f"anat/{ptID}_{ptSess}_FLAIR.nii.gz"),
                'dwi': join(src_path, f'ISLES-{version}RAW', ptID, ptSess, f"dwi/{ptID}_{ptSess}_dwi.nii.gz"),
                'adc': join(src_path, f'ISLES-{version}RAW', ptID, ptSess, f"dwi/{ptID}_{ptSess}_adc.nii.gz"),
                'segmentation': join(src_path, f'ISLES-{version}RAW', f"derivatives/{ptID}/{ptSess}/{ptID}_{ptSess}_msk.nii.gz")
            }

            for mod in keys:
                visit[mod+"_seg"] = visit[mod].replace(".nii.gz", "_seg.nii.gz")
            # print(visit)

            visit = transforms(visit)
            for k,v in visit.items():
                print(k, v.size())
            
            vol = torch.stack([visit[k].squeeze(0) for k in keys]).numpy()
            print("vol: ", vol.shape)
            np.save(join(tar_path, "vol", f"{ptID}_{ptSess}_vol.npy"), vol)
            
            seg = visit["segmentation"].squeeze(0).numpy()
            
            print("Seg: ", seg.shape)
            np.save(join(tar_path, "seg", f"{ptID}_{ptSess}_seg.npy"), seg)

            synthseg = torch.stack([visit[k].squeeze(0) for k in keys_seg if k != "segmentation"]).numpy()
            print("Synthseg: ", synthseg.shape)
            np.save(join(tar_path, "synthseg", f"{ptID}_{ptSess}_vol_synthseg.npy"), synthseg)

        # exit()



if __name__ == '__main__':
    dsetname = sys.argv[1]
    if 'brats' in dsetname.lower():
        preprocess_BRATS(dsetname)
    elif 'isles' in dsetname.lower():
        preprocess_ISLES(dsetname)
    elif 'glioma' in dsetname.lower():
        preprocess_GliomaPost(dsetname)
    else:
        raise NotImplementedError