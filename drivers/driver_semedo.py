from core.runtime import runtime
from core import constants
from methods import Semedo

# Define configurations to loop over
monkeys = constants.MONKEYS # ['Monkey N', 'Monkey F']
z_scores = sorted(constants.ZSCORE_INFO.keys()) # [1, 2, 3, 4]
analysis_types = constants.ANALYSIS_TYPES # ('window', 'baseline100', 'residual')

if __name__ == "__main__":
    print(f"Starting Batch Processing...")
    print(f"Monkeys: {monkeys}")
    print(f"Z-Scores: {z_scores}")
    print(f"Analysis Types: {analysis_types}")

    for monkey in monkeys:
        for z_score in z_scores:
            print(f"\n\n{'='*80}")
            print(f"CONFIGURATION: {monkey} | Z-Score Mode: {z_score}")
            print(f"{'='*80}")
            
            try:
                # 0. Configure Runtime
                runtime.update(monkey, z_score)
                
                for analysis_type in analysis_types:
                    print(f"\n{'-'*40}")
                    print(f"  Analysis Type: {analysis_type.upper()}")
                    print(f"{'-'*40}")
                    
                    try:
                        # 2. Subset Semedo (Figure 4 Subset)
                        print(f"  [1/5] Running Figure 4 Subset (V1 -> IT)...")
                        Semedo.build_figure_4_subset(
                            source_region=1, 
                            target_region=3, 
                            k_subsets=10, 
                            analysis_type=analysis_type,
                            force_recompute=False
                        )
                        print(f"  [2/5] Running Subset Semedo (V1 -> V4)...")
                        Semedo.build_figure_4_subset(
                            source_region=1, 
                            target_region=2, 
                            k_subsets=10, 
                            analysis_type=analysis_type,
                            force_recompute=False
                        )
                        
                        # 3. Regular Semedo (Figure 4)
                        print(f"  [3/5] Running Figure 4 (V1 -> IT)...")
                        Semedo.build_figure_4(
                            source_region=1, 
                            target_region=3, 
                            analysis_type=analysis_type,
                            force_recompute=False
                        )
                        # 3. Regular Semedo
                        print(f"  [4/5] Running Regular Semedo (V1 -> V4)...")
                        Semedo.build_figure_4(
                            source_region=1, 
                            target_region=2, 
                            analysis_type=analysis_type,
                            force_recompute=False
                        )
                        
                        # 4. Figure 5B
                        print(f"  [5/5] Running Figure 5B...")
                        Semedo.build_semedo_figure_5_b(
                            analysis_type=analysis_type,
                            force_recompute=False
                        )
                        
                        print(f"  >>> SUCCESS for {analysis_type} ({monkey}, Z={z_score})")

                    except Exception as inner_e:
                        print(f"  !!! ERROR in analysis {analysis_type}: {inner_e}")
                        continue

                print(f"\n>>> COMPLETED CONFIG: {monkey} | Z-Score {z_score}")
                
            except Exception as e:
                print(f"\n!!! CRITICAL ERROR in configuration {monkey} | Z-Score {z_score}:")
                print(f"{e}")
                continue

    print("\n\nAll Batch Processing Completed.")
