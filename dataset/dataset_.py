from dataset.UAVEOT import UAVEOT
from dataset.lasot import Lasot_lmdb, Lasot
from dataset.rseot import Rseot


def names2datasets(name_list: list, settings, image_loader, mode):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        assert name in ["RSEOT", "LASOT", "UAVEOT" ]
        if name == "LASOT":
            if settings.use_lmdb:
                print("Building lasot dataset from lmdb")
                datasets.append(Lasot_lmdb(settings.env.lasot_lmdb_dir, split='train', image_loader=image_loader))
            else:
                datasets.append(
                    Lasot(settings.dataset_path, split=mode, image_loader=image_loader, data_specs=settings.data_specs)
                )
        elif name == "RSEOT":
            datasets.append(
                Rseot(settings.dataset_path, split=mode, image_loader=image_loader, data_specs=settings.data_specs)
            )
        elif name == "UAVEOT":
            datasets.append(
                UAVEOT(settings.dataset_path, split=mode, image_loader=image_loader, data_specs=settings.data_specs)
            )
    return datasets