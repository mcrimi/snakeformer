import os

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    curriculum_file = os.path.join(base_dir, "snake_data_curriculum.txt")
    dagger_file = os.path.join(base_dir, "snake_curriculum_dagger_fixes.txt")
    output_file = os.path.join(base_dir, "snake_data.txt") # Standard training file name
    
    print(f"Merging {curriculum_file} and {dagger_file} into {output_file}...")
    
    line_count = 0
    with open(output_file, "w") as fout:
        # 1. Curriculum
        if os.path.exists(curriculum_file):
            print(f"Reading {curriculum_file}...")
            with open(curriculum_file, "r") as fin:
                for line in fin:
                    fout.write(line)
                    line_count += 1
        else:
            print(f"Warning: {curriculum_file} not found.")

        # 2. Dagger
        if os.path.exists(dagger_file):
            print(f"Reading {dagger_file}...")
            with open(dagger_file, "r") as fin:
                for line in fin:
                    fout.write(line)
                    line_count += 1
        else:
            print(f"Warning: {dagger_file} not found.")
            
    print(f"Done. Total lines: {line_count}")

if __name__ == "__main__":
    main()
