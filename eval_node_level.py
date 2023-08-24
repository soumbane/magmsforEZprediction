import torch
from ezpred import metrics
from magnet import Manager
from ezpred.configs import TestingConfigs
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torchmanager.metrics import metric
from torchmanager_core import view
from typing import Any
import numpy as np
import pandas as pd
import os

import data


def get_target_dict(num: int) -> dict[int, str]:

    if num == 1:
        dict_mod = {4:"DWIC"}
    elif num == 2:
        dict_mod = {3:"DWI"}
    elif num == 3:
        dict_mod = {3:"DWI", 4:"DWIC"}
    elif num == 4:
        dict_mod = {2:"FLAIR"}
    elif num == 5:
        dict_mod = {2:"FLAIR", 4:"DWIC"}
    elif num == 6:
        dict_mod = {2:"FLAIR", 3:"DWI"}
    elif num == 7:
        dict_mod = {2:"FLAIR", 3:"DWI", 4:"DWIC"}
    elif num == 8:
        dict_mod = {1:"T2"}
    elif num == 9:
        dict_mod = {1:"T2", 4:"DWIC"}
    elif num == 10:
        dict_mod = {1:"T2", 3:"DWI"}
    elif num == 11:
        dict_mod = {1:"T2", 3:"DWI", 4:"DWIC"}
    elif num == 12:
        dict_mod = {1:"T2", 2:"FLAIR"}
    elif num == 13:
        dict_mod = {1:"T2", 2:"FLAIR", 4:"DWIC"}
    elif num == 14:
        dict_mod = {1:"T2", 2:"FLAIR", 3:"DWI"}
    elif num == 15:
        dict_mod = {1:"T2", 2:"FLAIR", 3:"DWI", 4:"DWIC"}
    elif num == 16:
        dict_mod = {0:"T1"}
    elif num == 17:
        dict_mod = {0:"T1", 4:"DWIC"}
    elif num == 18:
        dict_mod = {0:"T1", 3:"DWI"}
    elif num == 19:
        dict_mod = {0:"T1", 3:"DWI", 4:"DWIC"}
    elif num == 20:
        dict_mod = {0:"T1", 2:"FLAIR"}
    elif num == 21:
        dict_mod = {0:"T1", 2:"FLAIR", 4:"DWIC"}
    elif num == 22:
        dict_mod = {0:"T1", 2:"FLAIR", 3:"DWI"}
    elif num == 23:
        dict_mod = {0:"T1", 2:"FLAIR", 3:"DWI", 4:"DWIC"}
    elif num == 24:
        dict_mod = {0:"T1", 1:"T2"}
    elif num == 25:
        dict_mod = {0:"T1", 1:"T2", 4:"DWIC"}
    elif num == 26:
        dict_mod = {0:"T1", 1:"T2", 3:"DWI"}
    elif num == 27:
        dict_mod = {0:"T1", 1:"T2", 3:"DWI", 4:"DWIC"}
    elif num == 28:
        dict_mod = {0:"T1", 1:"T2", 2:"FLAIR"}
    elif num == 29:
        dict_mod = {0:"T1", 1:"T2", 2:"FLAIR", 4:"DWIC"}
    elif num == 30:
        dict_mod = {0:"T1", 1:"T2", 2:"FLAIR", 3:"DWI"}
    elif num == 31:
        dict_mod = {0:"T1", 1:"T2", 2:"FLAIR", 3:"DWI", 4:"DWIC"}
    else:
        raise ValueError(f"num should be betwen 1 and 31, got {num}")

    return dict_mod


def test(cfg: TestingConfigs, /, target_dict: dict[int, str] = {0:'T1'}, node_num: int = 1) -> Any:
    # load dataset
    validation_dataset = data.DatasetEZ_NodeLevel(cfg.batch_size, cfg.data_dir, node_num=node_num, mode=data.EZMode.VALIDATE)  
    
    # load checkpoint
    if cfg.model.endswith(".model"):
        manager = Manager.from_checkpoint(cfg.model, map_location=cfg.device)
        assert isinstance(manager, Manager), "Checkpoint is not a valid `ezpred.Manager`."
    else:
        raise NotImplementedError(f"Checkpoint {cfg.model} is currently not supported.")
    
    # set up confusion metrics
    bal_acc_fn = metrics.BalancedAccuracyScore()
    conf_met_fn = metrics.ConfusionMetrics(2)
    manager.metric_fns.update({
        "val_bal_accuracy": bal_acc_fn,
        "conf_met": conf_met_fn
        })

    ## 0:T1, 1:T2, 2:FLAIR, 3:DWI, 4:DWIC
    # manager.target_dict = {
    #     0: "T1",
    #     1: "T2",
    #     2: "FLAIR",
    #     3: "DWI",
    #     4: "DWIC",
    # }

    manager.target_dict = target_dict

    # print(manager.target_dict)

    # print(f'The best accuracy on validation set occurs at {manager.current_epoch + 1} epoch number')

    # test checkpoint with validation dataset
    # summary: dict[str, Any] = manager.test(validation_dataset, show_verbose=cfg.show_verbose, device=cfg.device, use_multi_gpus=cfg.use_multi_gpus)
    summary: dict[str, Any] = manager.test(validation_dataset, show_verbose=False, device=cfg.device, use_multi_gpus=cfg.use_multi_gpus)
    if conf_met_fn.results is not None:
        summary.update({"conf_met": conf_met_fn.results})
    # view.logger.info(summary)
    
    # test checkpoint with testing dataset
    # summary: dict[str, Any] = manager.test(testing_dataset, show_verbose=cfg.show_verbose, device=cfg.device, use_multi_gpus=cfg.use_multi_gpus)
    # if conf_met_fn.results is not None:
    #     summary.update({"conf_met": conf_met_fn.results})
    # view.logger.info(summary)
    return summary['accuracy'], manager.target_dict


if __name__ == "__main__":
    configs = TestingConfigs.from_arguments()

    def get_list_of_node_nums():
        node_numbers_with_smote = [
            "1","2","3","5","6","11","12","13","14","17","18","19","20",
            "29","30","33","34","35","36","37","38","39","41","42","43","44","45",
            "46","47","48","49","50","51","52","53","54","55","56","57","58","59","60",
            "61","62","63","64","65","66","67","68","69","70","71","72","77","78","79","80",
            "81","83","84","85","86","87","88","89","90","91","92","93","94","95","96","97","98",
            "99","100","101","102","103","104","105","107","108","109","110","111","112",
            "113","120","121","122","123","124","125","126","127","128","129","130","131","132","133",
            "134","135","136","137","140","141","144","145","146","147","148","149","150",
            "151","152","154","155","156","157","158","159","160","162","163","164","165","166","167",
            "168","169","170","175","176","177","185","187","191","192","193","194","195",
            "196","197","198","199","200","201","202","204","205","208","209","210","211",
            "212","213","214","215","216","217","220","221","222","224","225","226","227","228","229",
            "230","231","232","233","234","235","238","239","240","241","243","244","245","246","247",
            "248","249","250","251","252","253","254","255","256","257","259","260","261","262","263",
            "264","275","287","288","289","290","291","292","294","295","296","297","298","299","300",
            "301","302","303","304","305","306","313","316","320","321","322","325","326","327",
            "331","332","334","335","336","337","338","339","340","341","343","346","349","352",
            "353","354","355","356","357","359","360","361","362","363","364","365","366","367","368","369",
            "370","371","372","373","374","375","376","377","378","381","382","383","384","385",
            "386","387","388","389","390","391","392","393","394","395","396","397","398","399","400",
            "401","402","403","404","405","406","407","408","409","410","411","412","413","414","415","416","417",
            "418","419","420","421","422","423","424","425","426","427","428","429","430","431","432",
            "433","434","435","436","437","438","439","440","441","442","443","444","445","446","447","448",
            "449","450","451","452","453","454","455","456","458","459","460","461","462","463","464","465",
            "466","467","468","469","470","471","472","473","474","475","476","477","478","479","500","501",
            "502","503","504","505","506","507","508","509","510","511","512","513","514","515","516","517",
            "518","519","520","521","522","524","525","526","529","530","531","532","533","534","535",
            "536","537","538","539","540","541","542","543","544","545","546","547","548","549","550","551","552",
            "553","554","555","556","557","558","559","560","561","562","563","564","565","566","567","568",
            "569","570","571","572","573","574","575","576","577","578","579","581","582","583","584",
            "585","586","587","588","589","590","591","592","593","594","595","596","597","598","599",
            "600","601","602","603","604","605","606","607","608","609","610","611","612","613","614","615","616","617","618","619","620","621","622","623","624","625","626","627","628","629","630","631","632","633","634","635","636","637","638","639","640","641","642","643","644","645","646","647","648","649","650","651","652","653","655","656","657","658","659","660","661","662","663","664","665","666","667","668",
            "669","670","671","672","673","674","675","676","677","678","679","680","681","682","683",
            "685","686","687","688","690","691","692","693","694","695","696","697","698","699","700","701",
            "702","703","704","705","706","707","708","709","710","711","712","713","714","715","716","717",
            "718","719","720","721","722","723","724","725","726","727","728","730","731","732","733",
            "735","736","737","738","739","740","741","742","743","744","745","746","747","748","749","750",
            "751","756","757","758","759","760","761","762","763","764","765","766","767","769","770",
            "771","776","777","778","779","780","781","782","783","784","785","786","787","788",
            "789","790","791","792","793","794","795","796","797","798","799","800","801","802","803","804",
            "805","806","807","808","809","810","811","812","813","816","817","818","819","820",
            "821","822","823","824","825","826","827","828","829","830","831","832","834","835","836",
            "837","838","839","841","842","843","844","845","846","847","848","849","850","851",
            "852","853","854","855","856","857","858","859","860","861","862","863","864","865","866",
            "867","868","869","870","871","872","873","874","875","876","877","878","879","880",
            "881","882","883","885","886","887","888","889","890","891","892","893","894","895",
            "896","897","898","899","900","901","902","903","904","905","906","907","908","909","910","911","912",
            "913","914","915","916","917","918","919","920","921","922","923","924","925","926","927","928","929",
            "930","931","932","933","934","935","936","937","938","939","940","941","942","943","944","945","946",
            "947","948","949","950","951","952","953","954","955","956","957","958","959","960","961",
            "962","963","964","965","966","968","969","970","971","973","974","975","976","977","978","979",
            "980","981","982","983"
        ]

        return node_numbers_with_smote

    # list of all 827 nodes for which SMOTE is possible (atleast 1 EZ)
    # node_numbers_with_smote = get_list_of_node_nums()
    # node_numbers_with_smote = ["1","2","3","948"]
    node_numbers_with_smote = ["912", "56", "551", "787", "555", "911"]

    dict_mod = get_target_dict(31)   

    accuracy_list = []
    node_num_list = []

    for i in node_numbers_with_smote: 
        acc, mod_dict = test(configs, target_dict=dict_mod, node_num=int(i))
        accuracy_list.append(acc)

        # print(f"Testing modality combination for Node {i}: {mod_dict}, accuracy is: {acc}\n")
        print(f"Node {i} test accuracy is: {acc}\n")
        node_num_list.append(i)    

    # dictionary of lists
    result_dict = {'Node number': node_num_list, 'Accuracy': accuracy_list}
    # print(result_dict)

    # Zip the data and sort by Accuracy in descending order
    sorted_data = sorted(zip(result_dict['Node number'], result_dict['Accuracy']), key=lambda x: x[1], reverse=True)

    # Create a new dictionary with sorted values
    sorted_dict = {
        'Node number': [item[0] for item in sorted_data],
        'Accuracy': [item[1] for item in sorted_data]
    }

    # print(sorted_dict)
        
    df = pd.DataFrame(sorted_dict)

    # saving the dataframe
    path = "/home/neil/Lab_work/Jeong_Lab_Multi_Modal_MRI/magmsEZpred/"
    save_path = os.path.join(path,"Node_Level_Results")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    filename = "Node_num_vs_acc_Fold1.csv"
    save_filepath = os.path.join(save_path, filename)
    
    df.to_csv(save_filepath, header=False, index=False)

    print(f"Total number of nodes: {len(accuracy_list)}")
    print(f"Final Testing modality combination mean is: {np.mean(accuracy_list)}")

    