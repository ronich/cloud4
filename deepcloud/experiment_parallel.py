import boto3, subprocess
import datetime
import time
import argparse
import fabric
from fabric.api import run, hide

def parseArgs():
    parser = argparse.ArgumentParser(description='Experiment specs')
    parser.add_argument('--instance_type', type=str, help='instance type')
    return  parser.parse_args()

class benchmark:
    def __init__(self, experiments, instance_type, image_id="ami-a3b3d4b5", key_name='dpcld_test1'):
        self.experiments = experiments
        self.instance_type = instance_type
        self.image_id = image_id
        self.key_name = key_name
        self.run_date = datetime.date.strftime(datetime.date.today(), '%Y%m%d')
        self.client = boto3.client('ec2')
        self.ec2 = boto3.resource('ec2')
        self.p_ids = []


    def setUpInstances(self):
        num_inst = len(experiments)

        response = self.client.run_instances(ImageId=self.image_id,
                                InstanceType=self.instance_type,
                                MinCount=num_inst,
                                MaxCount=num_inst,
                                KeyName=self.key_name,
                                IamInstanceProfile={
                                    'Arn':'arn:aws:iam::465729247037:instance-profile/S3_Admin_Access'}
                                )

        assert response.get('ResponseMetadata').get('HTTPStatusCode') == 200, "Request ended with an error (HTTPStatusCode != 200)"
        assert len(response.get('Instances')) == num_inst, "Number of instances launched is equal to specified"

        instance_ids = []
        instance_ids.extend([_.get('InstanceId') for _ in response.get('Instances')])

        waiter = self.client.get_waiter('instance_status_ok')
        waiter.wait(InstanceIds=instance_ids)

        print('Set up instances: {}'.format(' '.join(instance_ids)))

        self.instances = [self.ec2.Instance(_) for _ in instance_ids]

        #return [self.ec2.Instance(_) for _ in instance_ids]

    def configureInstances(self):
        [self.configureInstance(i, e) for i, e in zip(self.instances, self.experiments)]

    def configureInstance(self, instance, experiment):
        instance.create_tags(Tags=[{'Key':'Name', 'Value':'{}_{} experiment'.format(*experiment)}])
        fabric.api.env.host_string = 'ec2-user@{}'.format(instance.public_dns_name)
        fabric.api.env.key_filename = '~/.ssh/{}.pem'.format(self.key_name)
        fabric.api.env.disable_known_hosts
        fabric.api.hide('output')
        fabric.operations.put('./{}_{}.py'.format(*experiment), '.')
        run('sudo pip3 install git+git://github.com/fchollet/keras.git --upgrade')
        run('sudo pip3 install pydot')
        run('sudo pip3 install graphviz')

    def runExperiments(self):
        [self.runExperiment(i, e) for i, e in zip(self.instances, self.experiments)]

    def runExperiment(self, instance, experiment):
        fabric.api.env.host_string = 'ec2-user@{}'.format(instance.public_dns_name)
        run('mkdir -p logs')
        run('nohup python3 -u {0}_{1}.py \
        --run_date {2} --dataset {0} --architecture {1} --instance_type {3} \
        >logs/{2}_{0}_{1}_{3}.log 2>logs/{2}_{0}_{1}_{3}.err < /dev/null &'
            .format(*experiment, self.run_date, self.instance_type), pty=False)
        self.p_ids.append((instance.id, run('pgrep python3')))

    def getExperimentLogs(self):
        running_instances = self.instances

        while running_instances:
            for instance in self.instances:
                fabric.api.env.host_string = 'ec2-user@{}'.format(instance.public_dns_name)

                # this raises errors - didnt work as expected
                status = benchmark_td1.client.describe_instance_status(InstanceIds=[instance.id])\
                .get('InstanceStatuses')[0]\
                .get('InstanceState')\
                .get('Name')

                if status != 'running':
                    print('Instance {} not running'.format(instance.id))
                    running_instances.remove(instance)
                    continue

                self.rSync(instance.public_dns_name)
                self.syncWithS3()

                try:
                    run('pgrep python3')
                except:
                    print('Process on instance {} not found'.format(instance.id))
                    running_instances.remove(instance)
                    continue

                time.sleep(60)

            print('{}: synchronized logs from all instances'.format(datetime.datetime.now().isoformat()))
        else:
            print('All instances are down or finished their tasks')

    def rSync(self, public_dns):
        fabric.operations.local('rsync -aL -e "ssh -i ~/.ssh/{}.pem -o StrictHostKeyChecking=no" \
        --include="/home/ec2-user/logs/*" --include="*.log" --include="*.err" \
        --include="*.out" --include="*.png" --exclude="*" \
        ec2-user@{}:/home/ec2-user/logs/ \
        ./logs/'.format(self.key_name, public_dns))

    def syncWithS3(self):
        run('aws s3 sync /home/ec2-user/logs s3://deepcloud-logstash/ --exclude="*"\
        --include="*.log" --include="*.err" \
        --include="*.out" --include="*.png" --region "us-east-1"')

if __name__ == "__main__":
    experiments = [("mnist","kerasdef"),
                   ("mnist","custom"),
                   ("imdb","kerasdef"),
                   ("imdb","lstm_kerasdef"),
                   ("cifar","kerasdef"),
                   ("cifar","custom")
                   ]
    args = parseArgs()

    benchmark_td1 = benchmark(experiments=experiments,instance_type=args.instance_type)
    benchmark_td1.setUpInstances()
    benchmark_td1.configureInstances()
    benchmark_td1.runExperiments()
    benchmark_td1.getExperimentLogs()

    term_responses = [_.terminate() for _ in benchmark_td1.instances]

    print(term_responses)
