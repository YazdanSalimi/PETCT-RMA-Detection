from HybridMisalignmentDetect import petct_ram_detect

"""
Parameters:
    model_directory: Path to the folder containing the .pkl model files downloaded from the provided link.
    
    list_of_casenames: A list of case names, matching in size and order with 
                       both list_of_pet_segmentations and list_of_ct_segmentations. 
                       The generated report will use these names.
                       
    list_of_pet_segmentations: A list of file paths pointing to NIfTI images of segmentation masks 
                               generated on PET images.
                               
    list_of_ct_segmentations: A list of file paths pointing to NIfTI images of segmentation masks 
                              generated on CT images.
                              
    reference_labels (optional): Default is "none". If you wish to evaluate model performance on your 
                                 own labeled dataset, provide reference labels here. 
                                 Values: 0 for No RMA (Rotation Misalignment), 1 for RMA.
                                 
    results_folder: The path to the folder where the generated Excel files with results will be saved.
    
    segment_values: A dictionary defining the organ labels used in both PET and CT segmentation masks.
                    The PET and CT segmentation masks should be multi-label segmentations where:
                    - 0 represents the background.
                    - Organs should be mapped with the following default labels:
                        {
                            "Liver": 1,
                            "Spleen": 2,
                            "Lungs": 3,
                            "Heart": 4
                        }
                    You can modify this dictionary based on your data.

Example usage:

if __name__ == "__main__":
    petct_ram_detect(
        model_directory=model_directory, 
        list_of_casenames=list_of_casenames,
        list_of_pet_segmentations=list_of_pet_segmentations,
        list_of_ct_segmentations=list_of_ct_segmentations, 
        reference_labels=reference_labels,
        results_folder=results_folder,
        segment_values=segment_values,
    )
"""
