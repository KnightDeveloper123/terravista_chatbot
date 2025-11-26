// src/pages/ScheduleMeeting.jsx
import React from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import {
    Box,
    Button,
    Field,
    Fieldset,
    Flex,
    Input,
    Stack,
    Textarea,
} from '@chakra-ui/react';
import { useParams } from 'react-router';
import useFetch from '@/hooks/useFetch';

interface ScheduleMeetingForm {
    meeting_date: string;
    description?: string;
}


const ScheduleMeeting = () => {
    const { register, handleSubmit, formState: { errors, isSubmitting }, reset } = useForm<ScheduleMeetingForm>();
    const { request } = useFetch();
    const { userId } = useParams();

    // console.log("userId", userId);

    const onSubmit: SubmitHandler<ScheduleMeetingForm> = async (data) => {
        try {
            console.log(data);

        } catch (error) {
            console.log(error);
        }
    };

    return (
        <Flex alignItems={'center'} justifyContent={'center'} h={'100vh'} p={4}>
            <Box w={'max-content'} p={6} boxShadow="lg" borderRadius="md">
                <form onSubmit={handleSubmit(onSubmit)}>
                    <Fieldset.Root size="lg" maxW="md">
                        <Stack mb={4}>
                            <Fieldset.Legend>Schedule a Meeting</Fieldset.Legend>
                            <Fieldset.HelperText>
                                Fill out the details below to schedule your meeting.
                            </Fieldset.HelperText>
                        </Stack>

                        <Fieldset.Content>

                            {/* Meeting Date */}

                            <Field.Root>
                                <Field.Label>Meeting Date & Time</Field.Label>
                                <Input type="datetime-local" {...register('meeting_date', { required: 'Meeting date is required' })} />
                            </Field.Root>
                            {errors.meeting_date && <Fieldset.HelperText color="red.500">{errors?.meeting_date?.message}</Fieldset.HelperText>}

                        </Fieldset.Content>

                        {/* Optional: Duration */}
                        <Field.Root>
                            <Field.Label>Description</Field.Label>
                            <Textarea {...register('description', { min: { value: 10, message: 'Description must be at least 10 words' } })} />
                        </Field.Root>

                        <Button type="submit" alignSelf="flex-start" colorPalette="blue" mt={4} loading={isSubmitting}>Schedule Meeting</Button>
                    </Fieldset.Root>
                </form>
            </Box>
        </Flex>
    );
};

export default ScheduleMeeting;
