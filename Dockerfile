# Please choose your required Underworld/UWGeo image version here.
# Available images may be found at
#   https://hub.docker.com/r/underworldcode/underworld2/tags
#   https://hub.docker.com/r/underworldcode/uwgeodynamics/tags

## Need to test if 2.16 is compatible.
FROM underworldcode/underworld2:v2.16.0-ompi

# THE FOLLOWING WILL USUALLY NOT REQUIRE MODIFICATION.

# This command will copy in all the files in your repo.
COPY --chown=jovyan:users . /home/jovyan/community_model

# Set working directory to where we've put files. 
WORKDIR /home/jovyan/community_model

# Create symbolic link to documentation. 
RUN ln -s .. underworld_documentation
