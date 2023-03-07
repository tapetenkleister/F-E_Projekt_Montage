 # write detected_assembly_step, position and rotation to a txt.file for examination
detected_assembly_step = 'Pyramid Step 1'
max_similarity = 98
matrix_image_position = (4,6)
template_position_matrix = 180
with open('Images_Results/result.txt', 'w') as f:
        f.write(f'Detected assembly step: ||' + detected_assembly_step + '||')
        f.write(f' with '+str(max_similarity)+'%'+' similarity\n')
        f.write(f'Center position in x and y: '+str(matrix_image_position)+'\n')
        f.write(f'Rotation: '+str(template_position_matrix)+'°'+'\n')